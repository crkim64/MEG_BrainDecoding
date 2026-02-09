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

# Versatile Diffusion 경로 (환경에 맞게 수정)
sys.path.append('versatile_diffusion')

# BrainDiffuser 라이브러리
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

# ==========================================
# 1. 설정 및 파라미터
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data', help="Root data directory containing train/test/dictionary")
parser.add_argument("--image_root", default='./data', help="Root folder of raw images")
parser.add_argument("--out_dir", default='./data/extracted_features', help="Output directory")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--device", default='cuda', help="Device to use")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
# 2. 이미지 경로 복원용 Dictionary 로드
# ==========================================
DICT_PATH = os.path.join(args.data_dir, 'image_path_dictionary.h5')
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"🚨 Dictionary file not found: {DICT_PATH}\nRun 'preprocess_optimized.py' first.")

print(f"📚 Loading Image Path Dictionary from {DICT_PATH}...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    # byte string -> str 디코딩
    paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
    
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p

print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} unique images mapped.")

# ==========================================
# 3. Dataset 정의 (HDF5 기반)
# ==========================================
class ThingsHDF5Dataset(Dataset):
    def __init__(self, h5_path, image_root):
        self.h5_path = h5_path
        self.image_root = image_root
        
        # 메타데이터 로드
        with h5py.File(h5_path, 'r') as f:
            self.category_nr = f['category_nr'][:]
            self.exemplar_nr = f['exemplar_nr'][:] # int8
            self.length = len(self.category_nr)

        # 전처리: 224x224, [-1, 1] (CLIP Vision Standard)
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(), 
        ])

    def __len__(self):
        return self.length

    def _get_full_path(self, rel_path):
        """상대 경로를 절대 경로로 변환 및 보정"""
        # 1. 기본 경로
        full_path = os.path.join(self.image_root, rel_path)
        if os.path.exists(full_path): return full_path
        
        # 2. 경로 보정 (images_test_meg -> images_meg)
        fixed_path = rel_path.replace('images_test_meg', 'images_meg')
        full_path = os.path.join(self.image_root, fixed_path)
        if os.path.exists(full_path): return full_path

        # 3. 폴더 구조 직접 탐색 (Robust)
        filename = os.path.basename(fixed_path)
        if '_' in filename:
            # Common case: split by last underscore
            cat_guess = filename.rsplit('_', 1)[0]
            try_path = os.path.join(self.image_root, 'images_meg', cat_guess, filename)
            if os.path.exists(try_path): return try_path

            # Iterative split for complex category names
            parts = filename.split('_')
            for i in range(1, len(parts)):
                sub_cat = "_".join(parts[:i])
                try_path = os.path.join(self.image_root, 'images_meg', sub_cat, filename)
                if os.path.exists(try_path): return try_path
            
        return None

    def __getitem__(self, idx):
        cat = self.category_nr[idx]
        ex = self.exemplar_nr[idx]
        
        rel_path = IMAGE_MAP.get((cat, ex), None)
        
        found = False
        img = None
        
        if rel_path:
            full_path = self._get_full_path(rel_path)
            if full_path:
                try:
                    img = Image.open(full_path).convert('RGB')
                    found = True
                except:
                    pass
        
        if not found:
            # Placeholder (Black Image)
            img = Image.new('RGB', (512, 512)) 

        tensor_img = self.transform(img)
        tensor_img = tensor_img * 2 - 1  # [0,1] -> [-1, 1]
        
        return tensor_img, 1 if found else 0

# ==========================================
# 4. 모델 로드 (CLIP Vision)
# ==========================================
print("🚀 Loading Versatile Diffusion Model (CLIP Vision)...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth' 

if not os.path.exists(pth):
    raise FileNotFoundError(f"Checkpoint not found at {pth}")

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# 최적화: CLIP만 남기고 불필요한 모듈 제거
net.autokl = None
net.model = None

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device).float()
net.clip.eval() 


print(f"✅ CLIP Vision Model Ready on {device}")

# ==========================================
# 5. 추출 및 HDF5 저장 함수
# ==========================================
def extract_and_save_hdf5(dataset, save_path):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_samples = len(dataset)
    
    # Feature Dimensions
    # CLIP ViT-L/14의 경우: 257 tokens (1 CLS + 256 Patch), 768 dim
    n_tokens = 257
    n_dim = 768
    
    print(f"💾 Creating Feature HDF5: {save_path}")
    print(f"   Target Shape: ({num_samples}, {n_tokens}, {n_dim})")
    
    total_missing = 0
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # FP32으로 저장
        dset = f.create_dataset('features', shape=(num_samples, n_tokens, n_dim), dtype='float32', chunks=True)
        
        start_idx = 0
        with torch.no_grad():
            for batch_imgs, batch_valid_flags in tqdm(loader, desc=f"Extracting"):
                batch_imgs = batch_imgs.to(device, dtype=torch.float32)
                
                # Count missing
                missing_in_batch = (1 - batch_valid_flags).sum().item()
                total_missing += missing_in_batch
                
                # CLIP Encode Vision
                # BrainDiffuser는 [Batch, 257, 768] 형태의 전체 토큰을 사용함
                c = net.clip_encode_vision(batch_imgs)
                
                # CPU로 이동 및 저장
                c_np = c.cpu().numpy().astype(np.float32)
                
                batch_len = c_np.shape[0]
                dset[start_idx : start_idx + batch_len] = c_np
                start_idx += batch_len
    
    print(f"✅ Saved to {save_path}")
    if total_missing > 0:
        print(f"❌ WARNING: {total_missing} images were NOT FOUND and replaced with black images.")
    else:
        print(f"✨ Success: All {num_samples} images were found and processed.")

# ==========================================
# 6. 메인 실행 (Train & Test)
# ==========================================

# ------------------------------------------------
# 6-1. Combined Train Feature Extraction
# ------------------------------------------------
train_h5 = os.path.join(args.data_dir, 'train', 'combined_train.h5')
# 결과: ./data/extracted_features/combined_clip_vision_train.h5
train_out_path = os.path.join(args.out_dir, 'combined_clip_vision_train.h5')

if os.path.exists(train_h5):
    if not os.path.exists(train_out_path):
        print(f"\n🚀 Processing Combined Train Data...")
        train_dataset = ThingsHDF5Dataset(train_h5, args.image_root)
        extract_and_save_hdf5(train_dataset, train_out_path)
    else:
        print(f"\n⚠ Skipping Train: {train_out_path} already exists.")
else:
    print(f"\n🚨 Train HDF5 not found: {train_h5}")

# ------------------------------------------------
# 6-2. Per-Subject Test Feature Extraction
# ------------------------------------------------
subjects = ['P1', 'P2', 'P3', 'P4']

for sub in subjects:
    test_h5 = os.path.join(args.data_dir, 'test', f'{sub}_test.h5')
    
    # 결과: ./data/extracted_features/P1/P1_clip_vision_test.h5
    sub_out_dir = os.path.join(args.out_dir, sub)
    test_out_path = os.path.join(sub_out_dir, f'{sub}_clip_vision_test.h5')
    
    if os.path.exists(test_h5):
        if not os.path.exists(test_out_path):
            print(f"\n🚀 Processing Test Data for {sub}...")
            test_dataset = ThingsHDF5Dataset(test_h5, args.image_root)
            extract_and_save_hdf5(test_dataset, test_out_path)
        else:
            print(f"⚠ Skipping {sub}: {test_out_path} already exists.")
    else:
        print(f"🚨 Test HDF5 not found for {sub}: {test_h5}")

print("\n🎉 All CLIP Vision Extractions Finished!")