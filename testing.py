import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/LOLBlur.yml', help = 'Config file of testing')
args = parser.parse_args()

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
path_options = args.config
opt = parse(path_options)

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models
from utils.test_utils import *
from utils.device import get_device, get_map_location, is_cuda
from ptflops import get_model_complexity_info

# Set CUDA devices - use all available GPUs
if is_cuda():
    num_gpus = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

#parameters for saving model
PATH_MODEL= create_path_models(opt['save'])


def load_model(model, path_weights, rank=0, use_multi=False):

    map_location = get_map_location(rank)
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    weights = checkpoints['params']

    if rank == 0:
        macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
        print('Network complexity: ' ,macs, params)

    # Add 'module.' prefix when loading into DDP-wrapped model
    if use_multi:
        weights = {'module.' + key: value for key, value in weights.items()}

    model.load_state_dict(weights)
    if rank == 0:
        print('Loaded weights correctly')

    return model

def run_evaluation(rank, world_size):

    use_multi = is_cuda() and world_size > 1
    setup(rank, world_size=world_size)
    # LOAD THE DATALOADERS
    test_loader, _ = create_test_data(rank, world_size=world_size, opt = opt['datasets'])
    # DEFINE NETWORK
    model, _, _ = create_model(opt['network'], rank=rank, use_multi=use_multi)

    model = load_model(model, opt['save']['path'], rank=rank, use_multi=use_multi)
    metrics_eval = {}

    # Ensure all processes have reached this point (only for DDP)
    if is_cuda() and dist.is_initialized():
        dist.barrier()
    # eval phase
    model.eval()
    metrics_eval, _ = eval_model(model, test_loader, metrics_eval, rank=rank, world_size=world_size, eta = True)
    # Ensure all processes have reached this point (only for DDP)
    if is_cuda() and dist.is_initialized():
        dist.barrier()
    # print some results
    if rank==0:
        if type(next(iter(metrics_eval.values()))) == dict:
            for key, metric_eval in metrics_eval.items():
                print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
        else:
            print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
    cleanup()

def main():
    if is_cuda():
        world_size = torch.cuda.device_count()
        print(f'Running evaluation on {world_size} GPU(s)')
        mp.spawn(run_evaluation, args=(world_size,), nprocs=world_size, join=True)
    else:
        print(f'Running evaluation on: {get_device()}')
        run_evaluation(rank=0, world_size=1)

if __name__ == '__main__':
    main()
