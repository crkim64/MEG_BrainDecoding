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

# VD Libraries
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# 1. 설정 및 파라미터
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name")
parser.add_argument("--device", default='cuda', help="Main Device")
parser.add_argument("--vd_device", default='cuda:0', help="Device for Versatile Diffusion")
parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
args = parser.parse_args()

# 경로 설정
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
OUT_DIR = f'results/image_generation/{args.subject}_oracle'
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. 메타데이터 로드
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
else:
    raise FileNotFoundError("Dictionary file not found.")

# ==========================================
# 3. 모델 초기화 (Versatile Diffusion Only)
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

# Device Patch
if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device

sampler = DDIMSampler_VD(net)

# ==========================================
# 4. 데이터 로드
# ==========================================
print(f"📂 Loading Test Data Indices from {TEST_MEG_PATH}")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

# Select Random Samples
indices = np.linspace(0, len(test_cats)-1, args.num_samples, dtype=int)

print(f"✨ Start Oracle Generation for {len(indices)} images...")

# Image Transforms
transforms = tvtrans.Compose([
    tvtrans.Resize(224),
    tvtrans.CenterCrop(224),
    tvtrans.ToTensor(),
])

# For AutoKL (256x256 often preferred but VD usually takes 256 or 512, let's check config. 
# VD uses SD 1.4 derived VAE, usually 512x512. Let's try 512 for AutoKL and 224 for CLIP)
transforms_512 = tvtrans.Compose([
    tvtrans.Resize(512),
    tvtrans.CenterCrop(512),
    tvtrans.ToTensor(),
])

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    # Dummy Unconditional
    dummy_text = ''
    utx = net.clip_encode_text(dummy_text).to(args.vd_device).half()
    dummy_img = torch.zeros((1, 3, 224, 224)).to(args.vd_device).half()
    uim = net.clip_encode_vision(dummy_img).to(args.vd_device).half()

    for idx in indices:
        cat_id = test_cats[idx]
        ex_id = test_exs[idx]
        
        # Get Image Path
        rel_path = IMAGE_MAP.get((cat_id, ex_id))
        if not rel_path:
            print(f"Skipping {idx}: Path not found in dictionary.")
            continue
            
        full_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Skipping {idx}: File not found at {full_path}")
            continue
            
        # Load GT Image
        pil_img = Image.open(full_path).convert('RGB')
        
        # Save Original Copy for Comparison
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])
        pil_img.save(os.path.join(OUT_DIR, f"oracle_{idx:04d}_{safe_name}_GT.png"))
        
        print(f"[{idx}] Processing {safe_name}...")

        # -------------------------------------------------
        # EXTRACT ORACLE FEATURES (The "Upper Bound")
        # -------------------------------------------------
        
        # 1. AutoKL Latent (Oracle)
        # Input to VAE needs to be [-1, 1]
        img_512 = transforms_512(pil_img).unsqueeze(0).to(args.vd_device).half()
        img_512 = (img_512 * 2.0) - 1.0 
        
        # Encode
        encoder_posterior = net.autokl.encode(img_512)
        z_oracle = encoder_posterior.sample() * 0.18215 # Scale Factor for SD VAE
        
        # 2. CLIP Vision (Oracle)
        # Input to CLIP needs to be [0, 1]? No, standard normalization usually.
        # VD's clip_encode_vision handles pre-processing if given input 
        # But looking at library code, it calls self.clip.encode(vision).
        # We can pass raw tensor (1,3,224,224) in [-1, 1] usually for VD wrapper?
        # Let's check generate_images_VersatileDiffusionpy. 
        # It calls: uim = net.clip_encode_vision(dummy_img) where dummy_img is zeros.
        # Let's rely on net.clip_encode_vision to handle simple tensor.
        
        # VD expects [-1, 1] usually.
        img_224 = transforms(pil_img).unsqueeze(0).to(args.vd_device).half()
        img_224 = (img_224 * 2.0) - 1.0
        
        c_vision_oracle = net.clip_encode_vision(img_224).to(args.vd_device).half()
        
        # 3. Text (Oracle - using "an image of {name}" or similar?)
        # Since we don't have captions, let's just use Unconditional Text 
        # or maybe the class name if we can parse it.
        # For now, let's use Dummy Text to verify VISUAL reconstruction.
        c_text_oracle = utx 

        # -------------------------------------------------
        # GENERATE (Reconstruction)
        # -------------------------------------------------
        
        # Parameters
        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
        strength = 0.75
        scale = 7.5
        t_enc = int(strength * 50)
        
        # Stochastic Encode (starting from Oracle Latent)
        z_enc = sampler.stochastic_encode(z_oracle, torch.tensor([t_enc]).to(args.vd_device))
        
        # Decode
        # Mixing = 1.0 (Vision Only)
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, c_vision_oracle], # [Uncond, Cond]
            second_conditioning=[utx, c_text_oracle],  # [Uncond, Cond]
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=0.0 # 0.0 means First Ctype (Vision) is dominant? 
                            # Wait, mixed_ratio is for "mixed run". 
                            # Need to verify definition.
                            # In generate script: mixed_ratio=(1 - args.mixing) where mixing=0.4 (Vision vs Text).
                            # If we want Vision Only (mixing=1.0), then ratio = 0.
                            # VD Code: 
                            # if 0 < mixed_ratio < 1: mix
                            # else: select one.
                            # Usually 0 = First, 1 = Second?
                            # Let's check `lib/model_zoo/vd.py`.
                            # apply_model_dc -> forward_dc -> mixed_run_dc.
                            # It seems complex. Let's trust the generation script convention:
                            # 0.0 = TextOnly? 
                            # parser.add_argument("--mixing", ... help="0.0=TextOnly, 1.0=VisionOnly")
                            # script passes `mixed_ratio=(1 - args.mixing)`.
                            # So if mixing=1.0 (Vision), ratio=0.0.
                            # So ratio=0 means First Conditioning (Vision).
                            # Wait, prompt says: 0.0=TextOnly... 
                            # If mixing=0 -> ratio=1 -> Second (Text).
                            # If mixing=1 -> ratio=0 -> First (Vision).
                            # Correct.
            
        )
        
        # VAE Decode
        z = z.float()
        net.autokl.to(args.vd_device).float() 
        x = net.autokl_decode(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Save
        save_path = os.path.join(OUT_DIR, f"oracle_{idx:04d}_{safe_name}_Recon.png")
        img = tvtrans.ToPILImage()(x[0].cpu())
        img.save(save_path)
        print(f"  > Saved Reconstruction: {save_path}")

print("🎉 Oracle Test Finished!")
