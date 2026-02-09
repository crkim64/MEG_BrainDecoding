import sys
import os
import argparse
import numpy as np
import torch
import h5py
import PIL
from PIL import Image
from torchvision import transforms as tvtrans

# Versatile Diffusion 경로 설정 (환경에 맞게 수정)
sys.path.append('versatile_diffusion')

# Brain Module 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Brain Module
try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

# VD Libraries
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# 1. 설정 및 파라미터
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name (P1, P2, ...)")
parser.add_argument("--diff_strength", type=float, default=0.75, help="Denoising Strength (0.0 ~ 1.0)")
parser.add_argument("--scale", type=float, default=7.5, help="Guidance Scale")
parser.add_argument("--device", default='cuda', help="Main Device")
parser.add_argument("--vd_device", default='cuda:0', help="Device for Versatile Diffusion")
parser.add_argument("--bm_device", default='cuda:0', help="Device for Brain Modules")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
args = parser.parse_args()

# 경로 설정
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
POS_DIR = os.path.join(DATA_DIR, 'sensor_positions')
OUT_DIR = f'results/image_generation/{args.subject}_text_only'
os.makedirs(OUT_DIR, exist_ok=True)

# 체크포인트 - Text Only (Latest or Best)
CKPT_TEXT   = './checkpoints/clip_text/best_model.pth' 
STATS_PATH  = './checkpoints/clip_text/clip_text_stats.npz'

# ==========================================
# 2. 메타데이터 및 통계 로드
# ==========================================
print(f"📚 Loading Image Dictionary from {DICT_PATH}...")
IMAGE_MAP = {}
if os.path.exists(DICT_PATH):
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p
    print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} entries.")

print(f"📊 Loading Train Statistics from {STATS_PATH}...")
if os.path.exists(STATS_PATH):
    stats = np.load(STATS_PATH)
    train_mean = torch.from_numpy(stats['mean']).float().to(args.vd_device)
    train_std = torch.from_numpy(stats['std']).float().to(args.vd_device)
    print(f"✅ Statistics Loaded. Mean Shape: {train_mean.shape}")
else:
    raise FileNotFoundError(f"Stats file not found at {STATS_PATH}")

# ==========================================
# 3. 모델 로드 유틸리티
# ==========================================
def load_brain_module(ckpt_path, out_dim_clip, out_dim_mse, device):
    print(f"Loading BrainModule from {ckpt_path}...")
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
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

# ==========================================
# 4. 모델 초기화 (VD + Brain)
# ==========================================
print("🚀 Loading Versatile Diffusion...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'

if not os.path.exists(pth):
    raise FileNotFoundError(f"VD Checkpoint not found at {pth}")

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# GPU 할당 
net.clip.to(args.vd_device).half()   
net.autokl.to(args.vd_device).half() 
net.model.to(args.vd_device).half()  

if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device

sampler = DDIMSampler_VD(net)

# Load Brain Module (Text Only)
bm_text = load_brain_module(CKPT_TEXT, out_dim_clip=768, out_dim_mse=77*768, device=args.bm_device)

# ==========================================
# 5. 데이터 로드 및 생성
# ==========================================
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

print(f"✨ Start Text-Only Generation for {args.num_samples} images...")

# Dummy Placeholders
dummy_text_str = ''
dummy_img_tensor = torch.zeros((1, 3, 224, 224)).to(args.vd_device).half()

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    utx = net.clip_encode_text(dummy_text_str).to(args.vd_device).half()
    uim = net.clip_encode_vision(dummy_img_tensor).to(args.vd_device).half()

    for i in range(args.num_samples):
        # 1. Input Prep
        meg_tensor = torch.from_numpy(meg_data[i]).unsqueeze(0).float().to(args.bm_device)
        batch_obj = BatchObject(meg_tensor, sub_idx.to(args.bm_device), sensor_pos.to(args.bm_device))
        
        cat_id = test_cats[i]
        ex_id = test_exs[i]
        original_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        
        print(f"\nProcessing {i}/{args.num_samples}: {safe_name}")

        # 2. Text Prediction
        _, pred_text_flat = bm_text({'meg': meg_tensor}, batch_obj)
        pred_text_raw = pred_text_flat.to(args.vd_device) # (1, 59136)
        
        # 3. Z-Score Normalization SKIP (DEBUG)
        # Raw Prediction matches Train Stats closely (Mean ~0.001, Std ~0.03)
        # Renormalization distorts the signal (Range -0.5~0.5 -> -2.8~2.5).
        pred_text = pred_text_raw.reshape(1, 77, 768).half()
        
        # p_mean = pred_text_raw.mean()
        # p_std = pred_text_raw.std()
        # pred_norm = (pred_text_raw - p_mean) / (p_std + 1e-6)
        # pred_scaled = pred_norm * train_std + train_mean
        # pred_text = pred_scaled.reshape(1, 77, 768).half()

        # 4. Sampling (Text Only Mode)
        # Latent: Random Noise (since AutoKL is not used)
        # Vision: Dummy (uim)
        # Text: Predicted (pred_text)
        
        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
        t_enc = int(args.diff_strength * 50)
        
        # Start from pure noise? or Stochastic encode of random latent?
        # Standard T2I starts from gaussian noise.
        # But here we are "image decoding" -> actually "generating from text embedding".
        # So we can start from pure noise (t=Tet).
        # Wait, args.diff_strength is 0.75. This implies img2img usually.
        # But if we don't have an initial image latent (AutoKL), we should probably do full generation (t=0 to T)?
        # Or just start from random noise at t_enc?
        # If we start from random noise at t_enc, and t_enc < 50, we only denoise partially?
        # If we want full generation, we should use standard sample loop from T?
        # The original script uses `stochastic_encode(init_latent)`. 
        # If we use random latent as init, it's effectively random noise.
        
        init_latent = torch.randn(1, 4, 64, 64).to(args.vd_device).half()
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(args.vd_device))
        
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, uim],   # Vision: Uncond Only (Dummy)
            second_conditioning=[utx, pred_text], # Text: [Uncond, Cond]
            t_start=t_enc,
            unconditional_guidance_scale=args.scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=1.0 # 1.0 means Second Conditioning (Text) is dominant (from VD Check)
                            # Wait, let's verify mixed_ratio again. 
                            # In ablation we used `mixed_ratio=(1 - args.mixing)`.
                            # If mixing (Vision %) is 0 (Text Only), then mixed_ratio = 1.
                            # So 1.0 = Second (Text). Correct.
        )
        
        # VAE Decode
        z = z.float()
        net.autokl.to(args.vd_device).float() 
        x = net.autokl_decode(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Save
        save_filename = f"gen_{i:04d}_{safe_name}_TextOnly.png"
        save_path = os.path.join(OUT_DIR, save_filename)
        img = tvtrans.ToPILImage()(x[0].cpu())
        img.save(save_path)
        print(f"  > Saved: {save_filename}")

print("\n🎉 Text-Only Generation Finished!")
