import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append('versatile_diffusion')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

def main():
    device = 'cuda:1'
    print(f"🚀 Loading Versatile Diffusion on {device}...")
    
    # Load VD
    vd_path = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()('vd_noema')
    net = get_model()(cfgm)
    net.load_state_dict(torch.load(vd_path, map_location='cpu'), strict=False)
    
    net.clip.to(device) # Keep float32 for safety
    
    # 1. Check Text Embedding Stats
    print("\n🔍 Checking Text Embedding Stats...")
    texts = ["A photo of a dog", "A beautiful sunset", "Abstract geometric shapes"]
    
    for txt in texts:
        # clip_encode_text returns (1, 77, 768)
        emb = net.clip_encode_text(txt).to(device) 
        mean = emb.mean().item()
        std = emb.std().item()
        min_v = emb.min().item()
        max_v = emb.max().item()
        print(f"'{txt}':\n  Mean={mean:.4f}, Std={std:.4f}, Min={min_v:.4f}, Max={max_v:.4f}")

    # 2. Check Vision Embedding Stats
    print("\n🔍 Checking Vision Embedding Stats...")
    # Random Image (0-1)
    dummy_img = torch.rand(1, 3, 224, 224).to(device)
    # clip_encode_vision returns (1, 257, 768)
    vis_emb = net.clip_encode_vision(dummy_img).to(device)
    
    mean = vis_emb.mean().item()
    std = vis_emb.std().item()
    min_v = vis_emb.min().item()
    max_v = vis_emb.max().item()
    
    print(f"Random Image:\n  Mean={mean:.4f}, Std={std:.4f}, Min={min_v:.4f}, Max={max_v:.4f}")

if __name__ == "__main__":
    main()
