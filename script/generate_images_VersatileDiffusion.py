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
parser.add_argument("--mixing", type=float, default=0.4, help="Mixing Ratio (Vision vs Text). 0.0=TextOnly, 1.0=VisionOnly")
parser.add_argument("--scale", type=float, default=7.5, help="Guidance Scale")
parser.add_argument("--device", default='cuda', help="Main Device")
parser.add_argument("--vd_device", default='cuda:0', help="Device for Versatile Diffusion")
parser.add_argument("--bm_device", default='cuda:0', help="Device for Brain Modules")
args = parser.parse_args()

# 경로 설정
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
POS_DIR = os.path.join(DATA_DIR, 'sensor_positions')
OUT_DIR = f'results/image_generation/{args.subject}'
os.makedirs(OUT_DIR, exist_ok=True)

# 체크포인트 경로
CKPT_AUTOKL = './checkpoints/autokl_final/best_model.pth' # Scaled Latent Model
CKPT_VISION = './checkpoints/clip_vision/best_model.pth'  # Full Sequence Vision
CKPT_TEXT   = './checkpoints/clip_text/best_model.pth'     # Full Sequence Text

# ==========================================
# 2. 메타데이터 로드 (파일명 매핑용)
# ==========================================
print(f"📚 Loading Image Dictionary from {DICT_PATH}...")
IMAGE_MAP = {}
if os.path.exists(DICT_PATH):
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        # 바이트 문자열인 경우 디코딩
        paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
        
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p
    print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} entries.")
else:
    print("⚠️ Dictionary file not found. Filenames will be numeric only.")

# ==========================================
# 3. 모델 로드 유틸리티
# ==========================================
def load_brain_module(ckpt_path, out_dim_clip, out_dim_mse, device):
    """BrainModule을 초기화하고 가중치를 로드합니다."""
    print(f"Loading BrainModule from {ckpt_path}...")
    
    # 학습 때와 동일한 설정이어야 함
    model = BrainModule(
        in_channels={'meg': 271},
        out_dim_clip=out_dim_clip,
        out_dim_mse=out_dim_mse,
        time_len=281,
        hidden={'meg': 320},
        n_subjects=4, # P1~P4
        merger=True, merger_pos_dim=512, merger_channels=270,
        rewrite=True, glu=1, glu_context=1,
        skip=True, batch_norm=True, post_skip=True, scale=1.0,
        subject_layers=True
    ).to(device)
    
    # Merger Patch
    if model.merger:
        model.merger.position_getter.get_positions = lambda batch: batch.meg_positions.to(device)
        model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)

    # Load Weights
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
# 4. 모델 초기화
# ==========================================

# 4-1. Load Versatile Diffusion
print("🚀 Loading Versatile Diffusion...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'

if not os.path.exists(pth):
    raise FileNotFoundError(f"VD Checkpoint not found at {pth}")

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# GPU 할당 (메모리 최적화)
net.clip.to(args.vd_device).half()   # CLIP Encoder
net.autokl.to(args.vd_device).half() # AutoKL VAE
net.model.to(args.vd_device).half()  # UNet

# Patch for missing device attribute in library files (UNetModelVD)
if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device

sampler = DDIMSampler_VD(net)

# 4-2. Load Brain Modules
print("🚀 Loading Brain Modules...")
# AutoKL: Output 16384 (Flattened)
bm_autokl = load_brain_module(CKPT_AUTOKL, out_dim_clip=4096, out_dim_mse=16384, device=args.bm_device)

# Vision: Output 257*768 (Full Seq)
bm_vision = load_brain_module(CKPT_VISION, out_dim_clip=768, out_dim_mse=257*768, device=args.bm_device)

# Text: Output 77*768 (Full Seq)
bm_text = load_brain_module(CKPT_TEXT, out_dim_clip=768, out_dim_mse=77*768, device=args.bm_device)

# ==========================================
# 5. 데이터 로드 및 생성 루프
# ==========================================
# Sensor Positions 로드
pos_path = os.path.join(POS_DIR, f"sensor_positions_{args.subject}.npy")
if os.path.exists(pos_path):
    sensor_pos = np.load(pos_path)
else:
    print("⚠️ Sensor position file not found. Using random.")
    sensor_pos = np.random.rand(271, 2)
    
sensor_pos = torch.from_numpy(sensor_pos).float().unsqueeze(0) # (1, 271, 2)

# Subject Index Map
sub_map = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
sub_idx = torch.tensor([sub_map[args.subject]]).long()

# Test Data 로드
print(f"📂 Loading Test Data: {TEST_MEG_PATH}")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    meg_data = f['meg'][:]      # (N_test, 271, 281)
    test_cats = f['category_nr'][:]  # Category Info
    test_exs = f['exemplar_nr'][:]   # Exemplar Info
    
print(f"✨ Start Generation for {len(meg_data)} images...")

# Generate
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    for i in range(len(meg_data)):
        # -------------------------------------------------
        # 1. Prepare Input & Filename
        # -------------------------------------------------
        meg_tensor = torch.from_numpy(meg_data[i]).unsqueeze(0).float().to(args.bm_device)
        batch_obj = BatchObject(meg_tensor, sub_idx.to(args.bm_device), sensor_pos.to(args.bm_device))
        
        # 메타데이터 추출
        cat_id = test_cats[i]
        ex_id = test_exs[i]
        
        # 파일명 생성: gen_0000_cat1854_ex12_dog.png
        original_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        # 경로에서 파일명만 추출 (확장자 제거)
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        # 안전한 파일명으로 변환
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        
        save_filename = f"gen_{i:04d}_cat{cat_id}_ex{ex_id}_{safe_name}.png"
        save_path = os.path.join(OUT_DIR, save_filename)

        # -------------------------------------------------
        # 2. Predict Features (Brain Decoding)
        # -------------------------------------------------
        
        # (A) AutoKL Latent Prediction
        _, pred_latent_flat = bm_autokl({'meg': meg_tensor}, batch_obj)
        pred_latent = pred_latent_flat.reshape(1, 4, 64, 64)
        
        # [DEBUG FIX] Normalize predicted latent to N(0, 1)
        # The predicted latent has a negative shift (-0.3) which causes black images.
        pl_mean = pred_latent.mean()
        pl_std = pred_latent.std()
        print(f"[DEBUG] Original Latent Mean: {pl_mean:.4f} | Std: {pl_std:.4f}")
        
        pred_latent = (pred_latent - pl_mean) / pl_std
        print(f"[DEBUG] Normalized Latent Mean: {pred_latent.mean().item():.4f} | Std: {pred_latent.std().item():.4f}")

        init_latent = pred_latent.to(args.vd_device).half()

        # (B) CLIP Vision Prediction
        _, pred_vision_flat = bm_vision({'meg': meg_tensor}, batch_obj)
        pred_vision = pred_vision_flat.reshape(1, 257, 768).to(args.vd_device).half()

        # (C) CLIP Text Prediction
        _, pred_text_flat = bm_text({'meg': meg_tensor}, batch_obj)
        pred_text = pred_text_flat.reshape(1, 77, 768).to(args.vd_device).half()

        # -------------------------------------------------
        # 3. Versatile Diffusion Sampling
        # -------------------------------------------------
        
        # (A) Prepare Unconditional Conditioning
        dummy_text = ''
        utx = net.clip_encode_text(dummy_text).to(args.vd_device).half()
        
        dummy_img = torch.zeros((1, 3, 224, 224)).to(args.vd_device).half()
        uim = net.clip_encode_vision(dummy_img).to(args.vd_device).half()

        # (B) Stochastic Encode
        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
        t_enc = int(args.diff_strength * 50)
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(args.vd_device))
        
        # (C) Decode (Dual Guided)
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, pred_vision], 
            second_conditioning=[utx, pred_text],  
            t_start=t_enc,
            unconditional_guidance_scale=args.scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1 - args.mixing) 
        )
        
        # (D) VAE Decode
        # Force float32 for VAE decoding to avoid precision issues
        z = z.float()
        net.autokl.to(args.vd_device).float() 
        x = net.autokl_decode(z)
        
        # Debug Prints
        if i == 0:
            print(f"[DEBUG] Pred Vision Mean: {pred_vision.mean().item():.4f} | Std: {pred_vision.std().item():.4f}")
            print(f"[DEBUG] Pred Text Mean:   {pred_text.mean().item():.4f}   | Std: {pred_text.std().item():.4f}")
            print(f"[DEBUG] Pred Latent Mean: {pred_latent.mean().item():.4f} | Std: {pred_latent.std().item():.4f}")
            print(f"[DEBUG] Latent z Mean:    {z.mean().item():.4f}       | Std: {z.std().item():.4f}")
            print(f"[DEBUG] VAE Output x Min: {x.min().item():.4f} | Max: {x.max().item():.4f} | Mean: {x.mean().item():.4f}")

        # (E) Save Image
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        img = tvtrans.ToPILImage()(x[0].cpu())
        img.save(save_path)
        
        if i % 10 == 0:
            print(f"Generated {i}/{len(meg_data)}: {save_filename}")

print("🎉 All Generation Finished!")