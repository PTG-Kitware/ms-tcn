import torch
import numpy as np
import random

from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.actions = list(actions_dict.keys())
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        """Reads a list of the training video filenames 
        and shuffles them in ``self.list_of_examples``

        :param vid_list_file: Path to the training bundle file
        """
        file_ptr = open(vid_list_file, "r")
        self.list_of_examples = file_ptr.read().split("\n")[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        """Load the tensors of ``batch_size`` videos

        :param batch_size: Number of videos to grab in the batch

        :return: The features and targets of the batch
        """
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split(".")[0] + ".npy")
            file_ptr = open(self.gt_path + vid, "r")
            content = file_ptr.read().split("\n")[:-1]

            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            batch_input.append(features[:, :: self.sample_rate])
            batch_target.append(classes[:: self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(
            len(batch_input),
            np.shape(batch_input[0])[0],
            max(length_of_sequences),
            dtype=torch.float,
        )
        batch_target_tensor = torch.ones(
            len(batch_input), max(length_of_sequences), dtype=torch.long
        ) * (-100)
        mask = torch.zeros(
            len(batch_input),
            self.num_classes,
            max(length_of_sequences),
            dtype=torch.float,
        )
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, : np.shape(batch_input[i])[1]] = torch.from_numpy(
                batch_input[i]
            )
            batch_target_tensor[i, : np.shape(batch_target[i])[0]] = torch.from_numpy(
                batch_target[i]
            )
            mask[i, :, : np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes, np.shape(batch_target[i])[0]
            )

        return batch_input_tensor, batch_target_tensor, mask


class TemporalWindowBatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, batch_size):
        self.list_of_examples = list()
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.actions = list(actions_dict.keys())
        self.features_path = features_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        
        self.weights = np.ones(len(self.actions))
        
    def read_data(self, vid_list_file):
        with open(vid_list_file, "r") as file_ptr:
            self.list_of_examples = file_ptr.read().split("\n")[:-1]
        
        _input = []
        _target = []
        _weights = []
        for vid in self.list_of_examples:
            vid_name = vid.split(".")[0]
            features = np.load(f"{self.features_path}/{vid_name}.npy")
            with open(f"{self.gt_path}/{vid}", "r") as file_ptr:
                content = file_ptr.read().split("\n")[:-1]

            classes = np.zeros(min(np.shape(features)[1], len(content)))
            weights = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                c = content[i]
                classes[i] = self.actions_dict[c]
                weights[i] = self.class_weights[c]

            _input.append(features)
            _target.append(classes)
            _weights.append(weights)

        # FIll in ds
        feature_frames = np.concatenate(_input, axis=1).transpose()
        target_frames = np.concatenate(_target, axis=0, dtype=int, casting='unsafe')
        weights = np.concatenate(_weights, axis=0, dtype=np.float)

        self.ds = TensorDataset(torch.tensor(feature_frames),
                                torch.tensor(target_frames))

        self.sampler = WeightedRandomSampler(
                            weights=weights,
                            num_samples=len(self.ds),
                            replacement=True
                        )

        self.dl = DataLoader(self.ds, batch_size=self.batch_size,
                             sampler=self.sampler,
                             num_workers=0,
                             pin_memory=True, drop_last=True)

    def create_weights(self):
        """Create weights per action class based on the
        frequency of the class in the dataset
        """
        gt_rate = np.zeros(len(self.actions))
        for vid in self.list_of_examples:
            gt_fn = f"{self.gt_path}/{vid}"
            with open(gt_fn, "r") as gt_f:
                gts = gt_f.readlines()
            
            for gt in gts:
                action_id = self.actions.index(gt.strip())
                gt_rate[action_id] += 1
            
        class_weights = 1 / gt_rate
        self.class_weights = dict(zip(self.actions, class_weights))
    
    def next_batch(self, batch_size, temporal_window):
        """
        """
        # iterate on sampler
        idxs = self.dl._index_sampler()
        mask = np.zeros((targets.shape[0]))
        # output of sampler is the last index of the sample
        # grab index - temporal window: index as sample
        # gt corresponds to the sampler output


        



