import torch
import os
import argparse
import random

from eval import eval
from model import Trainer_pytorch
from data.dataset import PTG_Dataset


#####################
# Arguments
#####################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=1
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train")
# parser.add_argument('--dataset', default="gtea")
parser.add_argument("--split", default="1")
parser.add_argument("--batch_size", default=1000, type=int)
parser.add_argument("--num_workers", default=20, type=int)
parser.add_argument("--window_size", default=30, type=int)

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 204
lr = 0.0005
num_epochs = 200
smoothing_loss = 0.015
val_batch_size = 1000

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
save_dir = f"{output_dir}/{exp_name}_windows"

model_dir = f"{save_dir}/models/split_{args.split}"
results_dir = f"{save_dir}/results/split_{args.split}"

eval_output = f"{results_dir}/eval"

for output_d in [save_dir, model_dir, results_dir, eval_output]:
    if not os.path.exists(output_d):
        os.makedirs(output_d)

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
# Data
#####################
# Load training vidoes
with open(vid_list_file, "r") as train_f:
    train_videos = train_f.read().split("\n")[:-1]

# Load validation vidoes
with open(vid_list_file_val, "r") as val_f:
    val_videos = val_f.read().split("\n")[:-1]

# Load test videos
with open(vid_list_file_tst, "r") as test_f:
    test_videos = test_f.read().split("\n")[:-1]

# Trainer
trainer = Trainer_pytorch(
    num_stages, num_layers, num_f_maps, features_dim, 
    num_classes,actions_dict, gt_path, features_path, 
    sample_rate, args.window_size, val_videos, val_batch_size, smoothing_loss
)

#####################
# Train
#####################
if args.action == "train":
    train_dataset = PTG_Dataset(
        train_videos, num_classes, actions_dict, gt_path,
        features_path, sample_rate, 
        args.window_size
    )
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_dataset.weights, 
        len(train_dataset), 
        replacement=True, 
        generator=None
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    trainer.train(
        model_dir,
        train_dataloader,
        num_epochs=num_epochs,
        learning_rate=lr,
        device=device,
    )

if args.action == "predict":
    trainer.predict(
        results_dir,
        test_videos,
        model_dir,
        vid_list_file_tst,
        device,
    )

#####################
# Eval
#####################
if args.action == "eval":
    acc, recall, f1 = eval(
        test_videos,
        gt_path,
        results_dir,
        eval_output
    )

