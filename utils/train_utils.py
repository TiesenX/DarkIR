import torch
import torch.distributed as dist
import sys, os
from lpips import LPIPS
import numpy as np
sys.path.append('../losses')
sys.path.append('../data/dataset_reader/datapipeline')
from losses import *
from data.dataset_reader.datapipeline import CropTo4
from utils.device import get_device, get_backend, is_cuda

# NOTE: calc_SSIM and crop_to_4 are created per-device in functions below
# (module-level creation would pin them to CPU, breaking multi-GPU training)
crop_to_4 = CropTo4()

def setup(rank, world_size):
    if not is_cuda():
        # No DDP on macOS/CPU — single-process training
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = get_backend()
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
def cleanup():
    if not is_cuda():
        return
    dist.destroy_process_group()
    
def save_model(model, path):
    if is_cuda() and dist.get_rank() != 0:
        return
    torch.save(model.state_dict(), path)

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)

def train_model(epoch, model, optim, all_losses, train_loader, metrics, adapter = None, rank = None, logging_step = 10):
    '''
    It trains the model, returning the model, optim, scheduler and metrics dict
    '''
    outside_batch = None
    mean_metrics = {'train_loss': [], 'train_psnr': [], 'train_og_psnr': [], 'train_ssim':[]}
    dev = get_device(rank)
    calc_SSIM = SSIM(data_range=1.).to(dev)  # create on correct device per rank
    total_steps = len(train_loader)
    for i, (high_batch, low_batch) in enumerate(train_loader):

        # Move the data to the device (CUDA/MPS/CPU)
        high_batch = high_batch.to(dev)
        low_batch = low_batch.to(dev)

        optim.zero_grad()
        # Feed the data into the model
        if adapter: enhanced_batch = model(low_batch, use_adapter = adapter)#, side_loss = False, use_adapter = None)
        else: enhanced_batch = model(low_batch)
        # print(enhanced_batch)
        # calculate loss function to optimize
        optim_loss = calculate_loss(all_losses, enhanced_batch, high_batch, outside_batch)

        # Calculate loss function for the PSNR
        loss = torch.mean((high_batch - enhanced_batch)**2)
        og_loss = torch.mean((high_batch - low_batch)**2)

        # and PSNR (dB) metric
        psnr = 20 * torch.log10(1. / torch.sqrt(loss))
        og_psnr = 20 * torch.log10(1. / torch.sqrt(og_loss))

        #calculate ssim metric
        ssim = calc_SSIM(enhanced_batch, high_batch)
        # print(optim_loss)
        optim_loss.backward()
        optim.step()

        mean_metrics['train_loss'].append(optim_loss.item())
        mean_metrics['train_psnr'].append(psnr.item())
        mean_metrics['train_og_psnr'].append(og_psnr.item())
        mean_metrics['train_ssim'].append(ssim.item()) 
        
        if rank == 0:
            if i % logging_step == 0:
                print(f"Epoch {epoch + 1}, Step {i}/{total_steps}: Train Loss: {optim_loss.item():.4f}, Train PSNR: {psnr.item():.4f}, Train SSIM: {ssim.item():.4f}")
    metrics['train_loss'] = np.mean(mean_metrics['train_loss'])
    metrics['train_psnr'] = np.mean(mean_metrics['train_psnr'])
    metrics['train_og_psnr'] = np.mean(mean_metrics['train_og_psnr'])
    metrics['train_ssim'] = np.mean(mean_metrics['train_ssim'])
    
    return model, optim, metrics

def eval_one_loader(model, test_loader, metrics, largest_capable_size = 1500, adapter = None, rank=0):
    # In DDP, only rank 0 evaluates to avoid redundant computation
    # (test loader is not distributed — all ranks would see the same images)
    if rank != 0:
        metrics['valid_psnr'] = 0.
        metrics['valid_ssim'] = 0.
        metrics['valid_lpips'] = 0.
        imgs_dict = {}
        return metrics, imgs_dict

    dev = get_device(rank)
    calc_SSIM = SSIM(data_range=1.).to(dev)  # create on correct device
    calc_LPIPS = LPIPS(net='vgg', verbose=False).to(dev)  # create once, reuse across batches
    mean_metrics = {'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[]}
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            _, _, H, W = high_batch_valid.shape
            
            if H >= largest_capable_size or W>=largest_capable_size: # we need to make crops
                high_batch_valid_crop, low_batch_valid_crop = crop_to_4(high_batch_valid, low_batch_valid) # this returns a list of crops of size [1, 3, H//2, W//2]

                valid_loss_batch = 0
                valid_ssim_batch = 0
                valid_lpips_batch = 0              
                for high_crop, low_crop in zip(high_batch_valid_crop, low_batch_valid_crop):
                    high_crop = high_crop.to(dev)
                    low_crop = low_crop.to(dev)

                    enhanced_crop = model(low_crop)#, use_adapter = None)
                    # loss
                    valid_loss_batch += torch.mean((high_crop - enhanced_crop)**2)
                    valid_ssim_batch += calc_SSIM(enhanced_crop, high_crop)
                    valid_lpips_batch += calc_LPIPS(enhanced_crop, high_crop)
                
                #the final value of lpips and ssim will be the mean of all the crops
                valid_ssim_batch  = valid_ssim_batch / 4
                valid_lpips_batch = valid_lpips_batch / 4
                enhanced_batch_valid = enhanced_crop
                high_batch_valid = high_crop
                low_batch_valid = low_crop
                
            else: # If not, we process the image normally
                high_batch_valid = high_batch_valid.to(dev)
                low_batch_valid = low_batch_valid.to(dev)
                if adapter: enhanced_batch_valid = model(low_batch_valid, use_adapter = adapter)#, side_loss = False, use_adapter = None)
                else: enhanced_batch_valid = model(low_batch_valid)
                # loss
                valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
                valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
                valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
    
            mean_metrics['valid_psnr'].append(valid_psnr_batch.item())
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
    
    metrics['valid_psnr'] = np.mean(mean_metrics['valid_psnr'])
    metrics['valid_ssim'] = np.mean(mean_metrics['valid_ssim'])
    metrics['valid_lpips'] = np.mean(mean_metrics['valid_lpips'])
    # print(low_batch_valid.shape, torch.max(low_batch_valid))
    imgs_dict = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    return metrics, imgs_dict

def eval_model(model, test_loader, metrics, largest_capable_size = 1500, adapter = None, rank=None):
    '''
    This function runs over the multiple test loaders and returns the whole metrics.
    '''
    # if rank != 0:
    #     return None, None
    #first you need to assert that test_loader is a dictionary
    # print(test_loader)
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}
    # print(test_loader)
    if len(test_loader) > 1:
        all_metrics = {}
        all_imgs_dict = {}
        # print(test_loader)
        for key, loader in test_loader.items():
            # print(key, loader)
            all_metrics[f'{key}'] = {}
            metrics, imgs_dict = eval_one_loader(model, loader, all_metrics[f'{key}'], largest_capable_size = largest_capable_size, adapter = adapter, rank=rank)
            all_metrics[f'{key}'] = metrics
            all_imgs_dict[f'{key}'] = imgs_dict
        if rank == 0: print(all_metrics)
        return all_metrics, all_imgs_dict
    
    else:
        metrics, imgs_dict = eval_one_loader(model, test_loader['data'], metrics, largest_capable_size = largest_capable_size, adapter = adapter, rank=rank)
        return metrics, imgs_dict