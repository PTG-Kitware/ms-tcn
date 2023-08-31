import torch
import os
import argparse
import random

from eval import eval
from model import Trainer
from batch_gen import BatchGenerator


#####################
# Arguments
#####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train")
# parser.add_argument('--dataset', default="gtea")
parser.add_argument("--split", default="1")

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 204  # 2048
bz = 1
lr = 0.0005
num_epochs = 200
smoothing_loss = 0.015

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#    sample_rate = 2

#####################
# Filepaths
#####################
# Inputs
exp_name = "coffee_conf_10_hands_dist"
data_root = "/data/users/hannah.defazio/ptg_nas/data_copy/"
exp_data = f"{data_root}/TCN_data/{exp_name}"

vid_list_file = f"{exp_data}/splits/train_activity.split{args.split}.bundle"
vid_list_file_val = f"{exp_data}/splits/val.split{args.split}.bundle"
vid_list_file_tst = f"{exp_data}/splits/test.split{args.split}.bundle"

features_path = f"{exp_data}/features/"
gt_path = f"{exp_data}/groundTruth/"
mapping_file = f"{exp_data}/mapping.txt"

# Outputs
output_dir = f"/data/PTG/cooking/training/activity_classifier/TCN"

save_dir = f"{output_dir}/{exp_name}_val"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_dir = f"{save_dir}/models/split_{args.split}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

results_dir = f"{save_dir}/results/split_{args.split}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eval_output = f"{results_dir}/eval"
if not os.path.exists(eval_output):
    os.makedirs(eval_output)

#####################
# Labels
#####################
file_ptr = open(mapping_file, "r")
actions = file_ptr.read().split("\n")[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

#####################
# Train
#####################
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(
        num_classes, actions_dict, gt_path, features_path, sample_rate
    )
    batch_gen.read_data(vid_list_file)
    trainer.train(
        model_dir,
        batch_gen,
        num_epochs=num_epochs,
        batch_size=bz,
        learning_rate=lr,
        device=device,
        smoothing_loss=smoothing_loss,
        vid_list_file_val=vid_list_file_val
    )

if args.action == "predict":
    trainer.predict(
        model_dir,
        results_dir,
        features_path,
        vid_list_file_tst,
        num_epochs,
        actions_dict,
        device,
        sample_rate,
    )

#####################
# Eval
#####################
if args.action == "eval":
    acc, recall, f1 = eval(vid_list_file_tst, gt_path, results_dir, eval_output)
