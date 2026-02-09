import sys
import os
import argparse
import numpy as np
import torch
import h5py
import PIL
from PIL import Image
from torchvision import transforms as tvtrans

# Versatile Diffusion 경로 설정
sys.path.append('versatile_diffusion')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# 1. 설정 및 파라미터
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name")
parser.add_argument("--vd_device", default='cuda', help="Device for Versatile Diffusion")
parser.add_argument("--num_samples", type=int, default=2, help="Number of samples (2 required)")
args = parser.parse_args()

# 경로 설정
DATA_DIR = './data'
OUT_DIR = f'results/image_generation/{args.subject}_oracle_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

# Raw Image Info for GT
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')

# ==========================================
# 2. 로드 유틸리티
# ==========================================
print(f"📚 Loading Image Dictionary...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
    for c, e, p in zip(cats, exs, paths):
        # Store full entry for prompt extraction later
        IMAGE_MAP[(c, e)] = p

print("🚀 Loading Versatile Diffusion...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
# Filter unexpected keys if necessary
sd = {k: v for k, v in sd.items() if k in net.state_dict()}
net.load_state_dict(sd, strict=False)

net.clip.to(args.vd_device)
net.autokl.to(args.vd_device)
net.model.to(args.vd_device)
if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device
sampler = DDIMSampler_VD(net)

# ==========================================
# 3. 데이터 로드 및 생성
# ==========================================
print("📂 Loading Test Metadata...")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

# Select Samples (First 2 valid or random)
indices = [0, 1] 

print(f"✨ Start Oracle Ablation for {indices}...")

# Transforms
transforms_224 = tvtrans.Compose([
    tvtrans.Resize(224),
    tvtrans.CenterCrop(224),
    tvtrans.ToTensor(),
])
transforms_512 = tvtrans.Compose([
    tvtrans.Resize(512),
    tvtrans.CenterCrop(512),
    tvtrans.ToTensor(),
])

# Dummy Placeholders
dummy_text = ''
utx = net.clip_encode_text(dummy_text).to(args.vd_device)
dummy_img = torch.zeros((1, 3, 224, 224)).to(args.vd_device)
uim = net.clip_encode_vision(dummy_img).to(args.vd_device)

def get_prompt_from_path(path_str):
    if not path_str: return "object"
    filename = os.path.basename(path_str)
    parent_dir = os.path.basename(os.path.dirname(path_str))
    
    category = ""
    if "images" not in parent_dir and parent_dir != ".":
        category = parent_dir
    else:
        if '_' in filename:
            category = filename.rsplit('_', 1)[0]
        else:
            category = os.path.splitext(filename)[0]
    return category.replace('_', ' ')

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    for idx in indices:
        cat_id = test_cats[idx]
        ex_id = test_exs[idx]
        
        # Get Path
        rel_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        if rel_path == "unknown":
            print(f"Skipping {idx}, path unknown")
            continue

        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        
        full_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(full_path):
             print(f"Skipping {idx}, file not found")
             continue

        # Load Image
        gt_img = Image.open(full_path).convert('RGB')
        gt_img.save(os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_GT.png"))
        
        print(f"Processing {idx}: {safe_name}")

        # ---------------------------
        # Compute Features On-The-Fly
        # ---------------------------
        
        # 1. AutoKL (512x512 -> Latent)
        img_512 = transforms_512(gt_img).unsqueeze(0).to(args.vd_device)
        img_512 = (img_512 * 2.0) - 1.0
        z_gt_dist = net.autokl.encode(img_512)
        z_gt = z_gt_dist.sample() * 0.18215 # Explicit Scaling
        
        # 2. CLIP Vision (224x224 -> Vision Embedding)
        img_224 = transforms_224(gt_img).unsqueeze(0).to(args.vd_device)
        img_224 = (img_224 * 2.0) - 1.0 # [-1, 1] for VD wrapper
        c_vis = net.clip_encode_vision(img_224).to(args.vd_device)
        
        # 3. CLIP Text (Prompt -> Text Embedding)
        prompt = get_prompt_from_path(rel_path)
        print(f"  Prompt: {prompt}")
        c_text = net.clip_encode_text([prompt]).to(args.vd_device)

        # ---------------------------
        # Ablation Loop
        # ---------------------------
        conditions = {
            'All':      (z_gt, c_vis, c_text, 0.5), 
            'AutoKL':   (z_gt, uim,   utx,    0.5), # Dummy Context
            'Vision':   (None,     c_vis, utx,    1.0), # No Latent, Vision Only
            'Text':     (None,     uim,   c_text, 0.0), # No Latent, Text Only
            'Vision+Text': (None,  c_vis, c_text, 0.5), # No Latent, Vision + Text
        }
        
        for mode, (z_in, c_v, c_t, mix) in conditions.items():
            
            # Latent Prep
            sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
            
            if z_in is not None:
                # Reconstruction / Editing Mode
                t_enc = int(0.75 * 50) # strength 0.75
                z_enc = sampler.stochastic_encode(z_in, torch.tensor([t_enc]).to(args.vd_device))
            else:
                # Generation Mode (From Scratch)
                t_enc = 50 # Full steps
                # Just sample from N(0, I)
                z_enc = torch.randn(1, 4, 64, 64).to(args.vd_device)

            # Decode
            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uim, c_v],
                second_conditioning=[utx, c_t],
                t_start=t_enc,
                unconditional_guidance_scale=7.5,
                xtype='image',
                first_ctype='vision',
                second_ctype='prompt',
                mixed_ratio=(1.0 - mix) 
            )
            
            # VAE Decode
            z = z.float()
            net.autokl.to(args.vd_device).float() 
            x = net.autokl_decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            
            img = tvtrans.ToPILImage()(x[0].cpu())
            img.save(os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_{mode}.png"))
            print(f"  > Saved {mode} Ablation")

print("🎉 Oracle Ablation Finished!")
