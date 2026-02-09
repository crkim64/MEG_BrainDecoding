import sys
import os
import argparse
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

# Add VDVAE path
# Assumes 'vdvae' folder is at /home/kcr/BrainDiffuser_fMRI/brain-diffuser/vdvae
VDVAE_PATH = '/home/kcr/MEGBrainDecoding/vdvae'
sys.path.append(VDVAE_PATH)

# VDVAE Imports
from utils import *
from train_helpers import *
from model_utils import load_vaes
from data import set_up_data
# We need strictly what's required. 
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
parser.add_argument("--image_root", default='./data', help="Root folder of raw images")
parser.add_argument("--out_dir", default='./data/extracted_features', help="Output directory")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", default='cuda', help="Device to use")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
# 2. Image Path Dictionary
# ==========================================
DICT_PATH = os.path.join(args.data_dir, 'image_path_dictionary.h5')
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"🚨 Dictionary file not found: {DICT_PATH}")

print(f"📚 Loading Image Path Dictionary from {DICT_PATH}...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    paths = f['image_path'][:].astype(str)
    
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p
print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} unique images.")

# ==========================================
# 3. Dataset (Modified for VDVAE)
# ==========================================
class HDF5ImageDatasetVDVAE(Dataset):
    def __init__(self, h5_path, image_root):
        self.h5_path = h5_path
        self.image_root = image_root
        
        with h5py.File(h5_path, 'r') as f:
            self.category_nr = f['category_nr'][:]
            self.exemplar_nr = f['exemplar_nr'][:]
            self.length = len(self.category_nr)

        # VDVAE expects 64x64
        self.resize = T.Resize((64, 64), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self):
        return self.length

    def _get_full_path(self, rel_path):
        full_path = os.path.join(self.image_root, rel_path)
        if os.path.exists(full_path): return full_path
        
        fixed_path = rel_path.replace('images_test_meg', 'images_meg')
        full_path = os.path.join(self.image_root, fixed_path)
        if os.path.exists(full_path): return full_path

        filename = os.path.basename(fixed_path)
        if '_' in filename:
            category = filename.rsplit('_', 1)[0]
            try_path = os.path.join(self.image_root, 'images_meg', category, filename)
            if os.path.exists(try_path): return try_path
            
        return None

    def __getitem__(self, idx):
        cat = self.category_nr[idx]
        ex = self.exemplar_nr[idx]
        rel_path = IMAGE_MAP.get((cat, ex), None)
        
        img = None
        if rel_path:
            full_path = self._get_full_path(rel_path)
            
            # Fallback logic
            if not full_path:
                fname = os.path.basename(rel_path)
                if '_' in fname:
                    cat_guess = fname.rsplit('_', 1)[0]
                    fallback_path = os.path.join(self.image_root, 'images_meg', cat_guess, fname)
                    if os.path.exists(fallback_path):
                        full_path = fallback_path
            
            if full_path:
                try:
                    img = Image.open(full_path).convert('RGB')
                except:
                    pass
        
        if img is None:
            img = Image.new('RGB', (64, 64)) # Black image if missing

        # Preprocessing for VDVAE:
        img = self.resize(img)
        img_np = np.array(img).astype(np.float32) # (64, 64, 3) -> HWC
        img_tensor = torch.from_numpy(img_np)
        
        valid_flag = 1 if rel_path and full_path else 0
        return img_tensor, valid_flag

# ==========================================
# 4. Model Setup
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
# Set up data to get preprocess_mn
# We mock H.data_root to avoid loading valid data if possible, or just ignore return 
# But set_up_data calls imagenet64(H.data_root) which might fail if path doesn't exist.
# However, we only need 'preprocess_func' which is a closure.
# The 'imagenet64' function tries to load .npy files from data_root.
# We must avoid this if we don't have the data there.
# We can reimplement preprocess_func manually to avoid dependency on set_up_data!

# Re-implementing preprocess_func based on data.py to avoid loading ImageNet
print("🛠️  Setting up Preprocessing (VDVAE)...")
def get_preprocess_func(H):
    # From data.py for imagenet64
    shift = -115.92961967
    scale = 1. / 69.37404
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    
    # Tensorfy
    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    
    def preprocess_func(x):
        # x is [batch_tensor]
        inp = x[0].cuda(non_blocking=True).float()
        # inp is (B, C, H, W) in [0, 255]
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        return inp, out
    return preprocess_func

preprocess_fn = get_preprocess_func(H)
print("✅ Preprocessing Ready")

print("⌛ Loading VAE...")
ema_vae = load_vaes(H, logprint=print)
print(f"✅ VAE Loaded")

# ==========================================
# 5. Extraction Function
# ==========================================
def extract_and_save_hdf5_vdvae(dataset, save_path):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_samples = len(dataset)
    num_latents = 31 # VDVAE specific
    
    print(f"💾 Creating HDF5: {save_path}")
    
    # We don't know the exact flattened dimension beforehand efficiently without running one.
    # Let's run robust shape inference on the first batch.
    
    first_batch, _ = next(iter(loader))
    dummy_input, _ = preprocess_fn([first_batch])
    
    with torch.no_grad():
        activations = ema_vae.encoder.forward(dummy_input)
        _, stats = ema_vae.decoder.forward(activations, get_latents=True)
        # Calculate feature dim
        feat_dim = 0
        for i in range(num_latents):
            # stats[i]['z'] shape: (B, C, H, W)
            z = stats[i]['z']
            feat_dim += np.prod(z.shape[1:])
            
    print(f"   Calculated Feature Dim: {feat_dim}")
    
    with h5py.File(save_path, 'w') as f:
        # Using float16 to save space, assuming high dim
        dset = f.create_dataset('features', shape=(num_samples, feat_dim), dtype='float16', chunks=True)
        
        start_idx = 0
        total_missing = 0
        
        for batch_imgs, batch_valid_flags in tqdm(loader, desc="Extracting"):
            # batch_imgs: (B, 64, 64, 3)
            data_input, _ = preprocess_fn([batch_imgs]) # Moves to CUDA
            
            missing_in_batch = (1 - batch_valid_flags).sum().item()
            total_missing += missing_in_batch
            
            with torch.no_grad():
                activations = ema_vae.encoder.forward(data_input)
                _, stats = ema_vae.decoder.forward(activations, get_latents=True)
                
                batch_latent = []
                for i in range(num_latents):
                    z = stats[i]['z'] # (B, C, H, W)
                    z_flat = z.view(z.shape[0], -1).cpu().numpy()
                    batch_latent.append(z_flat)
                
                # Concat all layers: (B, Total_Dim)
                final_feat = np.hstack(batch_latent).astype(np.float16)
                
                batch_len = final_feat.shape[0]
                dset[start_idx : start_idx + batch_len] = final_feat
                start_idx += batch_len
                
    print(f"✅ Saved to {save_path}")
    if total_missing > 0:
        print(f"❌ WARNING: {total_missing} images were MISSING.")
    else:
        print(f"✨ Success: All processed.")

# ==========================================
# 6. Main Execution
# ==========================================

# 6-1. Combined Train
train_h5 = os.path.join(args.data_dir, 'train', 'combined_train.h5')
train_out_path = os.path.join(args.out_dir, 'combined_vdvae_train.h5')

if os.path.exists(train_h5):
    if not os.path.exists(train_out_path):
        print(f"\n🚀 Processing Combined Train Data...")
        dataset = HDF5ImageDatasetVDVAE(train_h5, args.image_root)
        extract_and_save_hdf5_vdvae(dataset, train_out_path)
    else:
        print(f"⚠ Skipping Train: {train_out_path} exists.")

# 6-2. Per-Subject Test
subjects = ['P1', 'P2', 'P3', 'P4']
for sub in subjects:
    test_h5 = os.path.join(args.data_dir, 'test', f'{sub}_test.h5')
    sub_out_dir = os.path.join(args.out_dir, sub)
    os.makedirs(sub_out_dir, exist_ok=True)
    test_out_path = os.path.join(sub_out_dir, f'{sub}_vdvae_test.h5')
    
    if os.path.exists(test_h5):
        if not os.path.exists(test_out_path):
            print(f"\n🚀 Processing Test Data for {sub}...")
            dataset = HDF5ImageDatasetVDVAE(test_h5, args.image_root)
            extract_and_save_hdf5_vdvae(dataset, test_out_path)
        else:
            print(f"⚠ Skipping {sub}: {test_out_path} exists.")

print("\n🎉 DONE!")
