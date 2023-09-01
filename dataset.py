import torch
import numpy as np
import random


class PTG_Dataset(torch.utils.data.Dataset):
    def __init__(self, videos, num_classes, actions_dict, gt_path, features_path, sample_rate, window_size,
                 transform=None, target_transform=None):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.dataset_size = -1


        input_frames_list = []
        target_frames_list = []
        for vid in videos:
            features = np.load(self.features_path + vid.split(".")[0] + ".npy")
            file_ptr = open(self.gt_path + vid, "r")
            content = file_ptr.read().split("\n")[:-1]

            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            input_frames_list.append(features[:, :: self.sample_rate])
            target_frames_list.append(classes[:: self.sample_rate])

        self.feature_frames = np.concatenate(input_frames_list, axis=1, dtype=np.single).transpose()
        self.target_frames = np.concatenate(target_frames_list, axis=0, dtype=int, casting='unsafe')

    
        self.dataset_size = self.target_frames.shape[0] - self.window_size

        # Get weights for sampler by inverse count.  
        # Weights represent the GT of the final frame of a window starting from idx
        class_name, counts = np.unique(self.target_frames, return_counts=True)
        class_weights =  1. / counts
        class_lookup = dict()
        for i, cn in enumerate(class_name):
            class_lookup[cn] = class_weights[i]
        self.weights = np.zeros((self.dataset_size))
        for i in range(self.dataset_size):
            self.weights[i] = class_lookup[self.target_frames[i+self.window_size]]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        features = self.feature_frames[idx:idx+self.window_size,:]
        target = self.target_frames[idx:idx+self.window_size]
        # mask = torch.ones((target.shape[0]), dtype=np.single)

        mask = torch.ones(
            self.num_classes,
            self.window_size,
            dtype=torch.float,
        )

        # TODO: Add Transforms/Augmentations
        return features, target, mask



# import torch
# import os
# import argparse
# import random


# #####################
# # Arguments
# #####################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 1538574472
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

# parser = argparse.ArgumentParser()
# parser.add_argument("--action", default="train")
# # parser.add_argument('--dataset', default="gtea")
# parser.add_argument("--split", default="1")
# parser.add_argument("--batch_size", default="10")
# parser.add_argument("--num_workers", default="0")
# parser.add_argument("--window_size", default="30")

# args = parser.parse_args()

# # use the full temporal resolution @ 15fps
# sample_rate = 1

# #####################
# # Filepaths
# #####################
# # Inputs
# exp_name = "coffee_conf_10_hands_dist"
# data_root = "/data/users/hannah.defazio/ptg_nas/data_copy/"
# exp_data = f"{data_root}/TCN_data/{exp_name}"

# features_path = f"{exp_data}/features/"
# gt_path = f"{exp_data}/groundTruth/"
# mapping_file = f"{exp_data}/mapping.txt"


# #####################
# # Labels
# #####################
# file_ptr = open(mapping_file, "r")
# actions = file_ptr.read().split("\n")[:-1]
# file_ptr.close()
# actions_dict = dict()
# for a in actions:
#     actions_dict[a.split()[1]] = int(a.split()[0])

# num_classes = len(actions_dict)

# vid_list_file = f"{exp_data}/splits/train_activity.split{args.split}.bundle"
# vid_list_file_val = f"{exp_data}/splits/val.split{args.split}.bundle"
# vid_list_file_tst = f"{exp_data}/splits/test.split{args.split}.bundle"


# dataset = PTG_Dataset(
#     vid_list_file, num_classes, actions_dict, gt_path, features_path, sample_rate, int(args.window_size)
# )

# dataset[len(dataset)-1]
# sampler = torch.utils.data.WeightedRandomSampler(dataset.weights, len(dataset), replacement=True, generator=None)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size), sampler=sampler,
#            num_workers=int(args.num_workers), pin_memory=True, drop_last=True)

# for item in dataloader:
#     break

# pass