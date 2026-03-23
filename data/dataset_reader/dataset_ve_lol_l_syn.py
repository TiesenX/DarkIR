import os
from glob import glob

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def main_dataset_ve_lol_l_syn(rank = 1,
                         test_path='../../data/datasets/VE-LOL-L/VE-LOL-L-Syn-Full',
                         train_path=None,
                         batch_size_train=4,
                         batch_size_test=1, 
                         verbose=False, 
                         cropsize=512, 
                         flips = None,
                         num_workers=1, 
                         crop_type='Random',
                         world_size = 1):
    
    PATH_VALID = test_path
    PATH_TRAIN = train_path
        
    # paths to the low-light and normal sets of images
    # VE-LOL-L structure: VE-LOL-L-Cap-Full/{VE-LOL-L-Cap-Low_test, VE-LOL-L-Cap-Normal_test, ...}
    low_test_dir = os.path.join(PATH_VALID, 'VE-LOL-L-Syn-Low_test')
    normal_test_dir = os.path.join(PATH_VALID, 'VE-LOL-L-Syn-Normal_test')
    
    list_blur_valid = sorted([os.path.join(low_test_dir, f) for f in os.listdir(low_test_dir) if not f.startswith('.')])
    list_sharp_valid = sorted([os.path.join(normal_test_dir, f) for f in os.listdir(normal_test_dir) if not f.startswith('.')])

    # check if all the image routes are correct
    check_paths([list_blur_valid, list_sharp_valid])

    if PATH_TRAIN:
        low_train_dir = os.path.join(PATH_TRAIN, 'VE-LOL-L-Syn-Low_train')
        normal_train_dir = os.path.join(PATH_TRAIN, 'VE-LOL-L-Syn-Normal_train')
        
        list_blur_train = sorted([os.path.join(low_train_dir, f) for f in os.listdir(low_train_dir) if not f.startswith('.')])
        list_sharp_train = sorted([os.path.join(normal_train_dir, f) for f in os.listdir(normal_train_dir) if not f.startswith('.')])
        check_paths([list_blur_train, list_sharp_train])

    if verbose:
        print('Images in the subsets:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))
        if PATH_TRAIN:
            print("    -Images in the PATH_LOW_TRAIN folder: ", len(list_blur_train))
            print("    -Images in the PATH_HIGH_TRAIN folder: ", len(list_sharp_train))

    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])
    else:
        flip_transform = None

    # Load the dataset
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)
    
    if PATH_TRAIN:
        train_dataset = MyDataset_Crop(list_blur_train, list_sharp_train, cropsize=cropsize,
                                    tensor_transform=tensor_transform, test=False, flips=flip_transform, crop_type=crop_type)

    train_loader = None
    if world_size > 1:
        samplers = []
        if PATH_TRAIN:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, rank=rank)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                                    num_workers=num_workers, pin_memory=True, drop_last=False, sampler=train_sampler)
            samplers.append(train_sampler)

        # Now we need to apply the Distributed sampler
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

        # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
        samplers.append(test_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:        
        if PATH_TRAIN:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                                    num_workers=num_workers, pin_memory=True, drop_last=False)
        # #Load the data loaders
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        samplers = None

    return train_loader, test_loader, samplers

if __name__ == '__main__':
    train_loader, test_loader, samplers = main_dataset_ve_lol_l()
    
