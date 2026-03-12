import fire

import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm
from options.options import parse

import torch
import torch.optim
import torch.multiprocessing as mp

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import init_wandb, logging_dict, create_path_models
from utils.train_utils import *
from utils.device import get_device, is_cuda, is_mps

torch.autograd.set_detect_anomaly(True)

def run_model(rank, world_size, path_options):
    """
    Each process (or the single process on MPS) parses the config itself,
    so it works with both mp.spawn (CUDA) and direct call (MPS/CPU).
    """
    # Parse config in each process — mp.spawn can't share globals
    opt = parse(path_options)

    # Set CUDA devices on Linux
    if sys.platform != 'darwin':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['device']['gpus'])

    # Setup model save paths
    PATH_MODEL, NEW_PATH_MODEL, BEST_PATH_MODEL = create_path_models(opt['save'])

    # Compute largest image size for eval cropping
    if opt['datasets']['train']['batch_size_train'] >= 8:
        largest_capable_size = opt['datasets']['train']['cropsize'] * opt['datasets']['train']['batch_size_train']
    else:
        largest_capable_size = 1500

    # LOAD THE DATALOADERS
    train_loader, test_loader, samplers = create_data(rank, world_size=world_size, opt = opt['datasets'])
    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, macs, params = create_model(opt['network'], rank=rank)

    # save this stats into opt to upload to wandb
    opt['macs'] = macs
    opt['params'] = params

    # Transfer learning: load pretrained weights only (no optimizer/epoch restore)
    model = load_pretrained(model, opt['network'].get('pretrained_weights', None), rank=rank)

    # define the optimizer
    optim, scheduler = create_optim_scheduler(opt['train'], model)

    # if resume load the weights (overwrites pretrained if resume_training: True)
    model, optim, scheduler, start_epochs = resume_model(model, optim, scheduler, path_model = PATH_MODEL,
                                                         rank = rank, resume=opt['network']['resume_training'])

    all_losses = create_loss(opt['train'], rank=rank)
    # INIT WANDB
    # init_wandb(rank, opt)
    best_psnr= 0
    for epoch in tqdm(range(start_epochs, opt['train']['epochs'])):

        start_time = time.time()
        metrics_train = {'epoch': epoch,'best_psnr': best_psnr}
        metrics_eval = {}

        # shuffle the samplers of each loader
        shuffle_sampler(samplers, epoch)
        # train phase
        model.train()
        model, optim, metrics_train = train_model(model, optim, all_losses, train_loader,
                                            metrics_train, rank = rank, logging_step = 25)
        # eval phase
        print("Running evaluation...")
        model.eval()
        metrics_eval, imgs_dict = eval_model(model, test_loader, metrics_eval, 
                                                    largest_capable_size=largest_capable_size, rank=rank)
        
        # print some results
        if rank==0:
            print(f"Epoch {epoch + 1} of {opt['train']['epochs']} took {time.time() - start_time:.3f}s\n")
            if type(next(iter(metrics_eval.values()))) == dict:
                for key, metric_eval in metrics_eval.items():
                    v_psnr, v_ssim, v_lpips = metric_eval['valid_psnr'], metric_eval['valid_ssim'], metric_eval['valid_lpips']
                    print(f' \t {key} --- PSNR: {v_psnr}, SSIM: {v_ssim}, LPIPS: {v_lpips}')
            else:
                ds_name = opt['datasets']['name']
                v_psnr, v_ssim, v_lpips = metrics_eval['valid_psnr'], metrics_eval['valid_ssim'], metrics_eval['valid_lpips']
                print(f' \t {ds_name} --- PSNR: {v_psnr}, SSIM: {v_ssim}, LPIPS: {v_lpips}')
        # Save the model after every epoch
        best_psnr = save_checkpoint(model, optim, scheduler, metrics_eval = metrics_eval, metrics_train=metrics_train, 
                                    paths = {'new':NEW_PATH_MODEL, 'best': BEST_PATH_MODEL}, rank=rank)

        # log into wandb if needed
        if opt['wandb']['init'] and rank == 0: wandb.log(logging_dict(metrics_train, metrics_eval, imgs_dict))
        #update scheduler
        scheduler.step()


def main(path_options='./options/train/LOLBlur.yml'):
    # Parse once here just to read device config and wandb setting
    opt = parse(path_options)

    if is_cuda():
        # Multi-GPU training with DDP on CUDA
        # Pass path_options so each spawned process can parse its own config
        world_size = len(opt['device']['ids'])
        # mp.spawn(run_model, args=(world_size, path_options), nprocs=world_size, join=True)
        run_model(rank=0, world_size=1, path_options=path_options)

    else:
        # Single-process training on MPS (macOS) or CPU
        print(f'Running single-process training on: {get_device()}')
        run_model(rank=0, world_size=1, path_options=path_options)

    if opt['wandb']['init']:
        wandb.finish()

if _name_ == '_main_':
    fire.Fire(main)
