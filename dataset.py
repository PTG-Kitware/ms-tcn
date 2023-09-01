import torch
import numpy as np
import random


class PTG_Dataset(torch.utils.data.Dataset):
    def __init__(self, vid_list_file, num_classes, actions_dict, gt_path, features_path, sample_rate, window_size,
                 transform=None, target_transform=None):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.dataset_size = -1

        # Load Vidoes
        file_ptr = open(vid_list_file, "r")
        videos = file_ptr.read().split("\n")[:-1]
        file_ptr.close()

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
        # set weights to 0 for frames before window length
        # so they don't get picked
        self.weights[:self.window_size] = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Grab a window of frames ending at ``idx``

        :param idx: The last index of the window

        :return: features, targets, and mask of the window
        """
        idx = idx + 1 # adjust to make sure we end at ``idx``
        
        features = self.feature_frames[idx-self.window_size:idx, :]
        target = self.target_frames[idx-self.window_size:idx]

        mask = torch.ones(
            self.num_classes,
            self.window_size,
            dtype=torch.float
        )

        # TODO: Add Transforms/Augmentations
        
        return features, target, mask

