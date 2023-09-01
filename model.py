import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ubelt as ub
from torch import optim
import copy
import numpy as np

from eval import eval


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss from the paper 
    `https://arxiv.org/pdf/1708.02002v2.pdf`
    """
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        self.ce = nn.CrossEntropyLoss(
            ignore_index=-100, 
            reduction="none"
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        # Check reduction option and return loss accordingly
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss

class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(
            num_blocks, num_layers, num_f_maps, dim, num_classes
        )
        self.loss_f = FocalLoss() #nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, 
              batch_size, learning_rate, device, smoothing_loss,
              vid_list_file_val):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # TODO: Make similar to every other training loop...ever
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.loss_f(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        batch_target.view(-1),
                    )
                    loss += smoothing_loss * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask[:, :, 1:]
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            
            # Save
            torch.save(
                self.model.state_dict(),
                f"{save_dir}/epoch-{str(epoch + 1)}.model",
            )
            torch.save(
                optimizer.state_dict(), f"{save_dir}/epoch-{str(epoch + 1)}.opt"
            )

            # Validation
            if epoch % 10 == 0:
                val_results = f"{save_dir}/val"
                if not os.path.exists(val_results):
                    os.makedirs(val_results)
                val_results_dir = f"{val_results}/epoch_{epoch+1}"
                if not os.path.exists(val_results_dir):
                    os.makedirs(val_results_dir)
                val_eval_results_dir = f"{val_results_dir}/eval"
                if not os.path.exists(val_eval_results_dir):
                    os.makedirs(val_eval_results_dir)

                self.predict(
                    save_dir,
                    val_results_dir,
                    batch_gen.features_path,
                    vid_list_file_val,
                    epoch+1,
                    batch_gen.actions_dict,
                    device,
                    batch_gen.sample_rate,
                )

                acc, recall, f1 = eval( vid_list_file_val,
                                        batch_gen.gt_path,
                                        val_results_dir,
                                        val_eval_results_dir
                                    )
            # Print
            print(
                "[epoch %d]: epoch loss = %f,   acc = %f,   val acc = %f"
                % (
                    epoch + 1,
                    epoch_loss / len(batch_gen.list_of_examples),
                    float(correct) / total,
                    acc
                )
            )

    def predict(
        self,
        model_dir,
        results_dir,
        features_path,
        vid_list_file,
        epoch,
        actions_dict,
        device,
        sample_rate,
    ):
        action_ids = list(actions_dict.values())
        action_strs = list(actions_dict.keys())

        self.model.eval()

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(
                torch.load(model_dir + "/epoch-" + str(epoch) + ".model")
            )
            file_ptr = open(vid_list_file, "r")
            list_of_vids = file_ptr.read().split("\n")[:-1]
            file_ptr.close()
            for vid in ub.ProgIter(list_of_vids, desc="Predicting videos"):
                features = np.load(features_path + vid.split(".")[0] + ".npy")
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device)
                )
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []

                for i in range(len(predicted)):
                    x = [action_strs[action_ids.index(predicted[i].item())]]
                    recognition = np.concatenate((recognition, x * sample_rate))
                f_name = vid.split("/")[-1].split(".")[0]
                f_ptr = open(results_dir + "/" + f_name + ".txt", "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(" ".join(recognition))
                f_ptr.close()
        print(f"Saved predictions to: {results_dir}")


class Trainer_pytorch:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, actions_dict):
        self.model = MultiStageModel(
            num_blocks, num_layers, num_f_maps, dim, num_classes
        )
        self.loss_f = FocalLoss() #nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes
        self.actions_dict = actions_dict

    def train(self, save_dir, train_dataloader, predict_dataloader, num_epochs, 
              learning_rate, device, smoothing_loss,
              vid_list_file_val):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            for batch_input, batch_target, mask in train_dataloader:
                batch_input = batch_input.transpose(2, 1)
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0

                for p in predictions:
                    loss += self.loss_f(
                        p[:,:,-1],
                        batch_target[:,-1]
                    )
                    loss += smoothing_loss * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask[:, :, 1:]
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total += torch.sum(mask[:, 0, :]).item()
                break
            
            # Save
            model_path = f"{save_dir}/epoch-{str(epoch + 1)}.model"
            torch.save(
                self.model.state_dict(),
                model_path,
            )
            torch.save(
                optimizer.state_dict(), f"{save_dir}/epoch-{str(epoch + 1)}.opt"
            )

            # Validation
            if epoch % 10 == 0:
                val_results = f"{save_dir}/val"
                if not os.path.exists(val_results):
                    os.makedirs(val_results)
                val_results_dir = f"{val_results}/epoch_{epoch+1}"
                if not os.path.exists(val_results_dir):
                    os.makedirs(val_results_dir)
                val_eval_results_dir = f"{val_results_dir}/eval"
                if not os.path.exists(val_eval_results_dir):
                    os.makedirs(val_eval_results_dir)

                self.predict(
                    predict_dataloader,
                    val_results_dir,
                    model_path,
                    device,
                )

                acc, recall, f1 = eval( vid_list_file_val,
                                        batch_gen.gt_path,
                                        val_results_dir,
                                        val_eval_results_dir
                                    )
            # Print
            print(
                "[epoch %d]: epoch loss = %f,   acc = %f,   val acc = %f"
                % (
                    epoch + 1,
                    epoch_loss,
                    float(correct) / total,
                    acc
                )
            )

    def predict(
        self,
        predict_dataloader,
        results_dir,
        model_path,
        device,
    ):
        action_ids = list(self.actions_dict.values())
        action_strs = list(self.actions_dict.keys())

        self.model.eval()

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(
                torch.load(model_path)
            )

            all_predictions = []
            for batch_input, batch_target, mask in predict_dataloader:
                batch_input = batch_input.transpose(2, 1)
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                predictions = self.model(batch_input, mask)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                    


            # for vid in ub.ProgIter(list_of_vids, desc="Predicting videos"):
            #     features = np.load(features_path + vid.split(".")[0] + ".npy")
            #     features = features[:, ::sample_rate]
            #     input_x = torch.tensor(features, dtype=torch.float)
            #     input_x.unsqueeze_(0)
            #     input_x = input_x.to(device)

            #     predictions = self.model(
            #         input_x, torch.ones(input_x.size(), device=device)
            #     )
            #     _, predicted = torch.max(predictions[-1].data, 1)
            #     predicted = predicted.squeeze()
            #     recognition = []

                # for i in range(len(predicted)):
                #     x = [action_strs[action_ids.index(predicted[i].item())]]
                #     recognition = np.concatenate((recognition, x * sample_rate))
                # f_name = vid.split("/")[-1].split(".")[0]
                # f_ptr = open(results_dir + "/" + f_name + ".txt", "w")
                # f_ptr.write("### Frame level recognition: ###\n")
                # f_ptr.write(" ".join(recognition))
                # f_ptr.close()
        print(f"Saved predictions to: {results_dir}")
