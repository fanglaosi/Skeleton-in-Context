import os
import sys
import argparse
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.utils.tools import get_config
from lib.data.dataset import MotionDataset3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file.")
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    if args.use_partial_data:
        args.data = args.partial_data
    else:
        args.data = args.full_data

    dataset = MotionDataset3D(args, data_split='train')
    query_target_all = []
    for i in range(len(dataset)):
        prompt_batch, query_batch, task_flag = dataset[i]
        query_target = query_batch[args.data.clip_len:]
        if args.flag_to_task[f'{task_flag}'] == 'MP' or args.flag_to_task[f'{task_flag}'] == 'MC':
            query_target[..., 1] = -query_target[..., 1]
        query_target_all.append(query_target)
    query_target_all = torch.stack(query_target_all)    # (N,T,17,3)
    query_target_avg = query_target_all.mean(0)     # (T,17,3)

    np.save(os.path.join(args.data.root_path, 'avg_pose.npy'), query_target_avg.data.numpy())