import sys
import os
import argparse
import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

# Add VDVAE path
# Assumes 'vdvae' folder is at /home/kcr/BrainDiffuser_fMRI/brain-diffuser/vdvae
# Or user corrected path: /home/kcr/MEGBrainDecoding/vdvae
# Let's check which one exists. Based on user edit, it's the local one.
VDVAE_PATH = '/home/kcr/MEGBrainDecoding/vdvae'
if not os.path.exists(VDVAE_PATH):
    # Fallback to the other one just in case
    VDVAE_PATH = '/home/kcr/BrainDiffuser_fMRI/brain-diffuser/vdvae'

sys.path.append(VDVAE_PATH)

# VDVAE Imports
from utils import *
from train_helpers import *
from model_utils import load_vaes

# H needs to be defined as dotdict
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# ==========================================
# 1. Configuration & Parameters
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data', help="Directory containing HDF5 data")
parser.add_argument("--subject", default='P1', help="Subject ID")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
parser.add_argument("--device", default='cuda', help="Device to use")
args = parser.parse_args()

OUT_DIR = f'results/image_generation/{args.subject}_oracle_ablation_extracted_VDVAE'
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. Loading Utilities
# ==========================================
DICT_PATH = os.path.join(args.data_dir, 'image_path_dictionary.h5')
IMAGE_MAP = {}
if os.path.exists(DICT_PATH):
    print(f"📚 Loading Image Dictionary...")
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p

# ==========================================
# 3. Model Setup
# ==========================================
print("🚀 Loading VDVAE Model...")

MODEL_ROOT = os.path.join(VDVAE_PATH, 'model')
# Hyperparams from reference script
H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 
     'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 
     'hparam_sets': 'imagenet64', 
     'restore_path': os.path.join(MODEL_ROOT, 'imagenet64-iter-1600000-model.th'), 
     'restore_ema_path': os.path.join(MODEL_ROOT, 'imagenet64-iter-1600000-model-ema.th'), 
     'restore_log_path': os.path.join(MODEL_ROOT, 'imagenet64-iter-1600000-log.jsonl'), 
     'restore_optimizer_path': os.path.join(MODEL_ROOT, 'imagenet64-iter-1600000-opt.th'), 
     'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 
     'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 
     'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 
     'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 
     'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 
     'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 
     'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 
     'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 
     'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 
     'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}

H = dotdict(H)

print("⌛ Loading VAE...")
ema_vae = load_vaes(H, logprint=print)
print(f"✅ VAE Loaded")

# ==========================================
# 4. Determine Latent Shapes (Dynamic)
# ==========================================
print("📏 Determining Latent Layer Shapes...")
# Dummy Run
dummy_input = torch.zeros(1, 64, 64, 3).float().cuda() # HWC
# Convert to CHW just like inside the model if needed?
# Wait, my previous discovery was that model expects HWC (Channels Last) input?
# Let's double check vae.py:
# `x = x.permute(0, 3, 1, 2).contiguous()` -> HWC to CHW.
# So input MUST be HWC.
# dummy_input is HWC.

layer_shapes = []
with torch.no_grad():
    activations = ema_vae.encoder.forward(dummy_input)
    # _, stats = ema_vae.decoder.forward(activations, get_latents=True) # Decoder requires activations
    # But for decoding from latents, we need the SHAPE of 'z' produced by stats.
    # The stats are produced by the posterior distribution in the decoder.
    # To mimic reconstruction, we need these shapes.
    
    # Let's run full forward pass to get stats structure
    _, stats = ema_vae.decoder.forward(activations, get_latents=True)
    
    num_latents_to_use = 31
    total_dim = 0
    for i in range(num_latents_to_use):
        # stats[i]['z'] shape: (B, C, H, W)
        z_shape = stats[i]['z'].shape[1:] # (C, H, W)
        layer_shapes.append(z_shape)
        # Flatten dim
        flat_dim = np.prod(z_shape)
        total_dim += flat_dim
        # print(f"Layer {i}: {z_shape} -> Flat: {flat_dim}")
    
print(f"✅ shapes determined. Total Flattened Dim: {total_dim}")
if total_dim != 91168:
    print(f"⚠️ WARNING: Calculated dim {total_dim} != 91168. Something might vary.")

# ==========================================
# 5. Load Features & Generate
# ==========================================
FEAT_PATH = os.path.join(args.data_dir, 'extracted_features', args.subject, f'{args.subject}_vdvae_test.h5')
TEST_META_PATH = os.path.join(args.data_dir, 'test', f'{args.subject}_test.h5')

print(f"📂 Loading Features from {FEAT_PATH}")
f_feat = h5py.File(FEAT_PATH, 'r')
features = f_feat['features']

print(f"📂 Loading Metadata from {TEST_META_PATH}")
with h5py.File(TEST_META_PATH, 'r') as f:
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

num_to_gen = min(args.num_samples, features.shape[0])
print(f"✨ Generating {num_to_gen} samples...")

for idx in range(num_to_gen):
    # 1. Get Flattened Features
    feat_flat = features[idx] # (91168,)
    
    # 2. Unflatten back to list of tensors
    # We slice the flat vector according to layer_shapes
    latents = []
    current_idx = 0
    
    for shape in layer_shapes:
        # shape is (C, H, W)
        flat_size = np.prod(shape)
        chunk = feat_flat[current_idx : current_idx + flat_size]
        current_idx += flat_size
        
        # Reshape to (1, C, H, W) and move to GPU
        tensor = torch.from_numpy(chunk).reshape(1, *shape).float().cuda()
        latents.append(tensor)
        
    # 3. Decode
    # vae.forward_samples_set_latents(n_batch, latents)
    with torch.no_grad():
        # returns (B, H, W, C) or (B, C, H, W)?
        # out_net.sample -> DmolNet.sample -> returns x_hat
        # Usually standard VAE returns images.
        # Let's check vae.py: forward_samples_set_latents calls decoder.forward_manual_latents -> final_fn
        # final_fn = lambda x: x * self.gain + self.bias
        # It operates on (N, C, H, W).
        # Wait, Block output is NCHW.
        # decoder returns xs[self.H.image_size].
        # So output is likely NCHW.
        
        # However, checking `out_net.sample(px_z)`.
        # `DmolNet` typically outputs distribution parameters. `sample` samples from it.
        # Images are typically RGB.
        
        
        img_recon = ema_vae.forward_samples_set_latents(1, latents)
        
        # img_recon is (1, 64, 64, 3) numpy array
        img_tensor = torch.from_numpy(img_recon[0]).float()
        
        print(f"[DEBUG] Raw Decoder Output Stats: Min={img_tensor.min():.4f}, Max={img_tensor.max():.4f}, Mean={img_tensor.mean():.4f}, Std={img_tensor.std():.4f}")

        # If mean is large (>10), it's likely already 0-255.
        if img_tensor.mean() < 10.0:
            print("[DEBUG] Applying Inverse Transform...")
            shift = -115.92961967
            scale = 1. / 69.37404
            
            # Simple inversion
            img_tensor = img_tensor / scale
            img_tensor = img_tensor - shift
        else:
             print("[DEBUG] Skipping Inverse Transform (Data seems already 0-255)")
        
        # Clip to 0-255
        img_tensor = torch.clamp(img_tensor, 0, 255)
        
        # HWC -> HWC (Already correct)
        img_np = img_tensor.cpu().numpy().astype(np.uint8)
        
        # Save Generated
        img_pil = Image.fromarray(img_np)
        
        # Metadata for filename
        cat_id = test_cats[idx]
        ex_id = test_exs[idx]
        rel_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        
        gen_path = os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_VDVAE_Recon.png")
        img_pil.save(gen_path)
        print(f"  > Saved: {gen_path}")
        
        # Helper: Save GT if available
        # (Copied from reference script logic)
        full_path = os.path.join(args.data_dir, rel_path)
        gt_path = os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_GT.png")
        if not os.path.exists(gt_path):
             if os.path.exists(full_path):
                 gt = Image.open(full_path).convert('RGB').resize((64, 64))
                 gt.save(gt_path)
             else:
                 # Try fallback
                 fname = os.path.basename(rel_path)
                 if '_' in fname:
                     cat_guess = fname.rsplit('_', 1)[0]
                     fallback_path = os.path.join(args.data_dir, 'images_meg', cat_guess, fname)
                     if os.path.exists(fallback_path):
                         gt = Image.open(fallback_path).convert('RGB').resize((64, 64))
                         gt.save(gt_path)

f_feat.close()
print("🎉 Done!")
