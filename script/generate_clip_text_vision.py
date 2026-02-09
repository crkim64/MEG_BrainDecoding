import sys
import os
import argparse
import numpy as np
import torch
import h5py
from torchvision import transforms as tvtrans
from PIL import Image

# Add paths
sys.path.append('versatile_diffusion')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# Configuration
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name")
parser.add_argument("--device", default='cuda:1', help="GPU Device")
parser.add_argument("--diff_strength", type=float, default=0.75, help="Strength of AutoKL init (1.0 = Pure Generation from Noise)")
parser.add_argument("--scale", type=float, default=7.5, help="Guidance Scale")
parser.add_argument("--mixing", type=float, default=0.6, help="Vision vs Text Mixing (0.0=TextOnly, 1.0=VisionOnly, 0.6=60% Vision)")
parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
args = parser.parse_args()

print(f"🚀 Settings: Device={args.device}, Subject={args.subject}, Strength={args.diff_strength}, Mixing={args.mixing}")

# Paths
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
# Checkpoints
# CKPT_AUTOKL Removed (Generating from Noise)
CKPT_VISION = './checkpoints/clip_vision/best_model.pth'
CKPT_TEXT = './checkpoints/clip_text/best_model.pth'

# ... (Output Dir Update)
OUT_DIR = f'results/image_generation/{args.subject}_no_autokl_no_scaling'
os.makedirs(OUT_DIR, exist_ok=True)




# ==========================================
# Helper Classes & Functions
# ==========================================
class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

def load_brain_module(ckpt_path, out_dim_clip, out_dim_mse, device):
    print(f"⌛ Loading BrainModule: {ckpt_path}")
    model = BrainModule(
        in_channels={'meg': 271},
        out_dim_clip=out_dim_clip,
        out_dim_mse=out_dim_mse,
        time_len=281,
        hidden={'meg': 320},
        n_subjects=4,
        merger=True, merger_pos_dim=512, merger_channels=270,
        rewrite=True, glu=1, glu_context=1,
        skip=True, batch_norm=True, post_skip=True, scale=1.0,
        subject_layers=True
    ).to(device)
    
    if model.merger:
        model.merger.position_getter.get_positions = lambda batch: batch.meg_positions.to(device)
        model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    else:
        print(f"⚠️ Checkpoint NOT FOUND: {ckpt_path}")
        return None

# ==========================================
# Main Execution
# ==========================================
def main():
    # 1. Load Dictionary
    image_map = {}
    if os.path.exists(DICT_PATH):
        with h5py.File(DICT_PATH, 'r') as f:
            cats = f['category_nr'][:]
            exs = f['exemplar_nr'][:]
            paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
            for c, e, p in zip(cats, exs, paths):
                image_map[(c, e)] = p

    # 2. Load Models
    # bm_autokl = load_brain_module(CKPT_AUTOKL, 4096, 16384, args.device) # Removed
    bm_vision = load_brain_module(CKPT_VISION, 768, 257*768, args.device)
    bm_text = load_brain_module(CKPT_TEXT, 768, 77*768, args.device)

    print("⌛ Loading Versatile Diffusion...")
    vd_path = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()('vd_noema')
    net = get_model()(cfgm)
    net.load_state_dict(torch.load(vd_path, map_location='cpu'), strict=False)
    
    # Move to GPU and Half Precision
    net.clip.to(args.device).half()
    net.autokl.to(args.device).half()
    net.model.to(args.device).half()
    if hasattr(net.model, 'diffusion_model'):
        net.model.diffusion_model.device = args.device
    net.model.device = args.device
    
    sampler = DDIMSampler_VD(net)

    # AutoKL VAE Scale Factor (Diffusion Std ~= 1.0)
    SCALE_FACTOR = 1.0 
    
    # CLIP Statistics (from debug_vd_stats.py)
    # Vision: Mean=0.0, Std=0.025
    # Text: Mean=0.0, Std=0.035
    VIS_MEAN, VIS_STD = 0.0, 0.025
    TXT_MEAN, TXT_STD = 0.0, 0.035

    # 3. Load Test Data
    print(f"📂 Loading MEG Data: {TEST_MEG_PATH}")
    with h5py.File(TEST_MEG_PATH, 'r') as f:
        meg_data = f['meg'][:args.n_samples]
        test_cats = f['category_nr'][:args.n_samples]
        test_exs = f['exemplar_nr'][:args.n_samples]

    # Sensor Positions
    pos_path = os.path.join(DATA_DIR, 'sensor_positions', f"sensor_positions_{args.subject}.npy")
    sensor_pos = np.load(pos_path) if os.path.exists(pos_path) else np.random.rand(271, 2)
    sensor_pos = torch.from_numpy(sensor_pos).float().unsqueeze(0).to(args.device)
    
    sub_idx = torch.tensor([{'P1':0, 'P2':1, 'P3':2, 'P4':3}[args.subject]]).long().to(args.device)

    # 4. Generation Loop
    print("✨ Starting Generation...")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(len(meg_data)):
            meg_tensor = torch.from_numpy(meg_data[i]).unsqueeze(0).float().to(args.device)
            batch = BatchObject(meg_tensor, sub_idx, sensor_pos)
            
            # (A) Predict Features
            # AutoKL
            # pred_lat_flat, _ = bm_autokl({'meg': meg_tensor}, batch) # Removed
            # pred_lat = pred_lat_flat.view(1, 4, 32, 32)
            
            # Fix: Normalize to correct VAE scale (0.18215)
            # Raw prediction might have low variance, so we force it to N(0, 0.18215)
            # pred_lat = (pred_lat - pred_lat.mean()) / (pred_lat.std() + 1e-6) * SCALE_FACTOR
            # init_lat = pred_lat.half()
            
            # if i == 0:
            #     print(f"[DEBUG] Net Scale Factor: {net.scale_factor if hasattr(net, 'scale_factor') else 'N/A'}")
            #     print(f"[DEBUG] Init Latent | Mean: {init_lat.mean().item():.4f}, Std: {init_lat.std().item():.4f}, Min: {init_lat.min().item():.4f}, Max: {init_lat.max().item():.4f}")
            #     if torch.isnan(init_lat).any(): print("🚨 NaN detected in Init Latent!")

            # CLIP Vision

            # CLIP Vision
            _, pred_vis_flat = bm_vision({'meg': meg_tensor}, batch)
            pred_vis = pred_vis_flat.view(1, 257, 768)
            
            if i == 0: print(f"[DEBUG] Raw Vision | Mean: {pred_vis.mean().item():.4f}, Std: {pred_vis.std().item():.4f}")
            
            # Fix: Normalize to correct CLIP scale (0.025)
            # pred_vis = (pred_vis - pred_vis.mean()) / (pred_vis.std() + 1e-6) * VIS_STD + VIS_MEAN
            pred_vis = pred_vis.half()
            
            if i == 0: print(f"[DEBUG] Norm Vision | Mean: {pred_vis.mean().item():.4f}, Std: {pred_vis.std().item():.4f}")

            # CLIP Text
            _, pred_txt_flat = bm_text({'meg': meg_tensor}, batch)
            pred_txt = pred_txt_flat.view(1, 77, 768)
            
            if i == 0: print(f"[DEBUG] Raw Text   | Mean: {pred_txt.mean().item():.4f}, Std: {pred_txt.std().item():.4f}")

            # Fix: Normalize to correct CLIP scale (0.035)
            # pred_txt = (pred_txt - pred_txt.mean()) / (pred_txt.std() + 1e-6) * TXT_STD + TXT_MEAN
            pred_txt = pred_txt.half()
            
            if i == 0: print(f"[DEBUG] Norm Text   | Mean: {pred_txt.mean().item():.4f}, Std: {pred_txt.std().item():.4f}")

            # (B) Diffusion
            sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
            
            # Start from pure noise (Strength = 1.0)
            t_enc = 50 
            z_enc = torch.randn((1, 4, 64, 64), device=args.device).half()

            # Unconditional Conditions
            u_img = net.clip_encode_vision(torch.zeros(1, 3, 224, 224).to(args.device).half())
            u_txt = net.clip_encode_text('').to(args.device).half()

            # Decode
            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[u_img, pred_vis],
                second_conditioning=[u_txt, pred_txt],
                t_start=t_enc,
                unconditional_guidance_scale=args.scale,
                xtype='image',
                first_ctype='vision',
                second_ctype='prompt',
                mixed_ratio=(1 - args.mixing) # 0.5 = Balanced
            )

            # (C) Decode to Image
            net.autokl.float()
            x = net.autokl_decode(z.float())
            x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
            img = tvtrans.ToPILImage()(x[0].cpu())

            # Save
            cat, ex = test_cats[i], test_exs[i]
            orig_path = image_map.get((cat, ex), 'unknown')
            base = os.path.splitext(os.path.basename(orig_path))[0]
            safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base])
            
            fname = f"gen_{i:03d}_cat{cat}_ex{ex}_{safe_name}.png"
            img.save(os.path.join(OUT_DIR, fname))
            
            print(f"Generated {i+1}/{len(meg_data)}: {fname} | Vis Mean: {pred_vis.mean():.4f}, Txt Mean: {pred_txt.mean():.4f}")

    print(f"🎉 Done! Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
