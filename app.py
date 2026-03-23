from torch._C._jit_tree_views import NoneLiteral
import gradio as gr
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from archs import create_model, load_pretrained
from options.options import parse

# ── Config ────────────────────────────────────────────────────────────────────
# Point this to any train YAML that matches the architecture of your checkpoint.
# Only the [network] section is used here; paths/datasets are ignored.
PATH_OPTIONS  = './options/train/LoLI_Street.yaml'

# Path to your trained checkpoint (.pt).
# Supports both save_checkpoint format (model_state_dict key)
# and external checkpoints (params key).
PATH_CHECKPOINT = '/Users/tienlm/Documents/Master/Code/AAI/darkir-materials/output/models/DarkIR-m-VELOLCAP-best.pt'
# PATH_CHECKPOINT = '/Users/tienlm/Documents/Master/Code/darkir-materials/models/DarkIR_384.pt'
# ──────────────────────────────────────────────────────────────────────────────

opt    = parse(PATH_OPTIONS)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Build model (rank=0, no DDP for demo)
model, _, _ = create_model(opt['network'], rank=0, use_multi=False)
model       = load_pretrained(model, PATH_CHECKPOINT, rank=0, use_multi=False, from_checkpoint=True)
model       = model.to(device)
model.eval()
print(f'Model loaded from {PATH_CHECKPOINT} on {device}')

pil_to_tensor = transforms.ToTensor()

def pad_to_multiple(tensor, multiple=8):
    """Pad H and W to be divisible by `multiple`."""
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h)), H, W

def process_img(image):
    """Run DarkIR on a PIL image and return restored PIL image."""
    if image is None:
        return None
    img = np.array(image.convert('RGB')).astype(np.float32) / 255.
    y   = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    y_padded, H, W = pad_to_multiple(y)
    with torch.no_grad():
        out = model(y_padded)

    # Unpad and convert back to PIL
    out = out[:, :, :H, :W].clamp(0, 1).squeeze(0)
    restored = (out.permute(1, 2, 0).cpu().numpy() * 255.).round().astype(np.uint8)
    return Image.fromarray(restored)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
title = "DarkIR — Low-Light Image Restoration 🌙✨"
description = """
## DarkIR: Robust Low-Light Image Restoration (CVPR 2025)

Upload a **low-light or blurry dark image** and DarkIR will restore it.

> Model: **DarkIR** — handles noise, blur, and low illumination in a single pass.  
> High-resolution images (>2K) may be slow or run out of memory.
"""

# Use teaser images from the repo as built-in examples
examples = []
for f in ['assets/teaser/0085_low.png', 'assets/teaser/low00747.png']:
    if os.path.exists(f):
        examples.append([f])

demo = gr.Interface(
    fn          = process_img,
    inputs      = gr.Image(type='pil', label='Input (low-light image)'),
    outputs     = gr.Image(type='pil', label='Restored output'),
    title       = title,
    description = description,
    examples    = examples if examples else None,
    css         = ".image-frame img { width: auto; height: auto; max-width: none; }"
)

if __name__ == '__main__':
    demo.launch(share=True)