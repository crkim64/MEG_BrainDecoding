import sys
import os
import argparse
import numpy as np
import torch
import h5py
from PIL import Image
from tqdm import tqdm

# Add Paths
sys.path.append('vdvae') 
# Brain Module Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports
try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

# VDVAE Imports
from utils import *
from train_helpers import *
from model_utils import load_vaes

# H needs to be defined as dotdict for VDVAE
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# ==========================================
# 1. Config & Args
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject ID")
parser.add_argument("--device", default='cuda:0', help="Device for Brain Module")
parser.add_argument("--vdvae_device", default='cuda:0', help="Device for VDVAE") # Can be same
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
args = parser.parse_args()

# Paths
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
POS_DIR = os.path.join(DATA_DIR, 'sensor_positions')
OUT_DIR = f'results/image_generation/{args.subject}_VDVAE'
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATH = './checkpoints/vdvae/best_model.pth'

# ==========================================
# 2. Load Metadata
# ==========================================
print(f"📚 Loading Image Dictionary...")
IMAGE_MAP = {}
if os.path.exists(DICT_PATH):
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p

# ==========================================
# 3. Load Brain Module
# ==========================================
print(f"🧠 Loading BrainModule from {CKPT_PATH}...")
model = BrainModule(
    in_channels={'meg': 271},
    out_dim_clip=768,      # Must match training config even if unused
    out_dim_mse=91168,     # VDVAE Features
    time_len=281,
    hidden={'meg': 320},
    n_subjects=4, 
    merger=True, merger_pos_dim=512, merger_channels=270,
    rewrite=True, glu=1, glu_context=1,
    skip=True, batch_norm=True, post_skip=True, scale=1.0,
    subject_layers=True
).to(args.device)

if model.merger:
    model.merger.position_getter.get_positions = lambda batch: batch.meg_positions.to(args.device)
    model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)

if os.path.exists(CKPT_PATH):
    sd = torch.load(CKPT_PATH, map_location=args.device)
    model.load_state_dict(sd, strict=False)
    model.eval()
else:
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

# ==========================================
# 4. Load VDVAE
# ==========================================
print("🚀 Loading VDVAE Model...")
# Setup VDVAE Hparams (Same as extraction)
VDVAE_PATH = 'vdvae' # Assumes accessible via sys.path
MODEL_ROOT = os.path.join(VDVAE_PATH, 'model')
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

ema_vae = load_vaes(H, logprint=print)
# Move to device if not handled by load_vaes (it uses H.device if set, usually defaults to cuda)
# We need to make sure input tensors are on same device.

# Determine Latent Shapes (Dynamic)
print("📏 Determining Latent Layer Shapes...")
dummy_input = torch.zeros(1, 64, 64, 3).float().cuda() # HWC
layer_shapes = []
with torch.no_grad():
    activations = ema_vae.encoder.forward(dummy_input)
    _, stats = ema_vae.decoder.forward(activations, get_latents=True)
    num_latents_to_use = 31
    total_dim = 0
    for i in range(num_latents_to_use):
        z_shape = stats[i]['z'].shape[1:] # (C, H, W)
        layer_shapes.append(z_shape)
        total_dim += np.prod(z_shape)

print(f"✅ shapes determined. Total Flattened Dim: {total_dim}")
assert total_dim == 91168

# ==========================================
# 5. Prediction & Generation Loop
# ==========================================
# Load Sensor Pos
pos_path = os.path.join(POS_DIR, f"sensor_positions_{args.subject}.npy")
if os.path.exists(pos_path):
    sensor_pos = np.load(pos_path)
else:
    sensor_pos = np.random.rand(271, 2)
sensor_pos = torch.from_numpy(sensor_pos).float().unsqueeze(0) 

sub_map = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
sub_idx = torch.tensor([sub_map[args.subject]]).long()

print(f"📂 Loading Test Data: {TEST_MEG_PATH}")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    meg_data = f['meg'][:]      
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

num_to_gen = min(args.num_samples, len(meg_data))
print(f"✨ Generating {num_to_gen} samples...")

for i in range(num_to_gen):
    # 1. Prediction
    meg_tensor = torch.from_numpy(meg_data[i]).unsqueeze(0).float().to(args.device)
    batch_obj = BatchObject(meg_tensor, sub_idx.to(args.device), sensor_pos.to(args.device))
    
    with torch.no_grad():
        _, pred_mse = model({'meg': meg_tensor}, batch_obj)
    
    # 1.1 Inverse Normalization (Z-score -> Raw)
    # Load Stats
    stats_path = './checkpoints/vdvae/vdvae_stats.npz'
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        t_mean = stats['mean']
        t_std = stats['std']
        
        # pred_flat is (91168,)
        pred_flat = pred_flat * (t_std + 1e-6) + t_mean
    else:
        print("⚠️ Warning: Stats file not found. Skipping inverse norm (Might produce garbage images).")
    
    # 2. Unflatten to Latents
    latents = []
    current_idx = 0
    for shape in layer_shapes:
        flat_size = np.prod(shape)
        chunk = pred_flat[current_idx : current_idx + flat_size]
        current_idx += flat_size
        tensor = torch.from_numpy(chunk).reshape(1, *shape).float().cuda()
        latents.append(tensor)
        
    # 3. Decode
    with torch.no_grad():
        img_recon = ema_vae.forward_samples_set_latents(1, latents)
        # img_recon is (1, 64, 64, 3) numpy array usually (or Tensor if changed?)
        # Let's handle numpy output
        if isinstance(img_recon, np.ndarray):
             img_tensor = torch.from_numpy(img_recon[0]).float()
        else:
             img_tensor = img_recon[0].float()
        
        # Check range
        if img_tensor.mean() < 10.0:
            # Needs Inverse Transform
            shift = -115.92961967
            scale = 1. / 69.37404
            img_tensor = img_tensor / scale
            img_tensor = img_tensor - shift
            
        img_tensor = torch.clamp(img_tensor, 0, 255)
        img_np = img_tensor.cpu().numpy().astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # 4. Save
        cat_id = test_cats[i]
        ex_id = test_exs[i]
        original_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        
        gen_path = os.path.join(OUT_DIR, f"{i:04d}_{safe_name}_Pred_VDVAE.png")
        img_pil.save(gen_path)
        print(f"  > Saved: {gen_path}")
        
        # Save GT if available (Copied logic)
        gt_path = os.path.join(OUT_DIR, f"{i:04d}_{safe_name}_GT.png")
        if not os.path.exists(gt_path):
             full_path = os.path.join(DATA_DIR, original_path)
             if os.path.exists(full_path):
                 Image.open(full_path).convert('RGB').resize((64, 64)).save(gt_path)

print("🎉 VDVAE Generation Finished!")
