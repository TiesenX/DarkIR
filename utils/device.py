"""
Centralized device detection for cross-platform PyTorch training.
Supports CUDA (Linux/Windows), MPS (macOS Apple Silicon), and CPU fallback.
"""
import sys
import torch


def get_device(rank=None):
    """
    Returns the appropriate torch.device for the current platform.
    
    Priority: CUDA > MPS > CPU
    
    Args:
        rank: GPU rank for multi-GPU CUDA setups. Ignored on MPS/CPU.
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if rank is not None:
            return torch.device(f'cuda:{rank}')
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_backend():
    """
    Returns the appropriate distributed backend for the current platform.
    - 'nccl' for CUDA (Linux)
    - 'gloo' for macOS / CPU (gloo supports MPS and CPU)
    """
    if torch.cuda.is_available():
        return 'nccl'
    else:
        return 'gloo'


def get_map_location(rank=0):
    """
    Returns the map_location argument for torch.load, platform-aware.
    
    Args:
        rank: target GPU rank (used for CUDA multi-GPU mapping)
    Returns:
        dict or torch.device for use with torch.load(..., map_location=...)
    """
    if torch.cuda.is_available():
        return {'cuda:%d' % 0: 'cuda:%d' % rank}
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def is_cuda():
    """Returns True if CUDA is available."""
    return torch.cuda.is_available()


def is_mps():
    """Returns True if MPS (Apple Silicon GPU) is available."""
    return torch.backends.mps.is_available()


def is_macos():
    """Returns True if running on macOS."""
    return sys.platform == 'darwin'
