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
from utils.device import get_device, is_cuda
from ptflops import get_model_complexity_info

# Set CUDA device only if available
if is_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

#parameters for saving model
PATH_MODEL= create_path_models(opt['save'])


def load_model(model, path_weights):

    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    weights = checkpoints['params']

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print('Network complexity: ' ,macs, params)

    model.load_state_dict(weights)
    print('Loaded weights correctly')

    return model

def run_evaluation(rank, world_size):

    setup(rank, world_size=world_size)
    # LOAD THE DATALOADERS
    test_loader, _ = create_test_data(rank, world_size=world_size, opt = opt['datasets'])
    # DEFINE NETWORK
    model, _, _ = create_model(opt['network'], rank=rank)

    model = load_model(model, opt['save']['path'])
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
    world_size = 1
    if is_cuda():
        mp.spawn(run_evaluation, args =(world_size,), nprocs=world_size, join=True)
    else:
        print(f'Running evaluation on: {get_device()}')
        run_evaluation(rank=0, world_size=1)

if __name__ == '__main__':
    main()
