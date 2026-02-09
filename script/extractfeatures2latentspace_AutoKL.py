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
parser.add_argument("--data_dir", default='./data', help="Directory containing HDF5 data")
parser.add_argument("--image_root", default='./data', help="Root folder of raw images")
parser.add_argument("--out_dir", default='./data/extracted_features', help="Output directory")
parser.add_argument("--batch_size", type=int, default=32)
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
    paths = f['image_path'][:].astype(str) # byte string -> str 변환
    
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p

print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} unique images.")

# ==========================================
# 3. Dataset 정의 (HDF5 기반)
# ==========================================
class HDF5ImageDataset(Dataset):
    def __init__(self, h5_path, image_root):
        self.h5_path = h5_path
        self.image_root = image_root
        
        # 메타데이터 미리 로드
        with h5py.File(h5_path, 'r') as f:
            self.category_nr = f['category_nr'][:]
            self.exemplar_nr = f['exemplar_nr'][:]
            self.length = len(self.category_nr)

        # 전처리: 512x512, [-1, 1]
        self.transform = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), 
        ])

    def __len__(self):
        return self.length

    def _get_full_path(self, rel_path):
        """상대 경로를 절대 경로로 변환 및 보정"""
        # 1. 기본 경로 시도
        full_path = os.path.join(self.image_root, rel_path)
        if os.path.exists(full_path): return full_path
        
        # 2. images_test_meg -> images_meg 수정 시도
        fixed_path = rel_path.replace('images_test_meg', 'images_meg')
        full_path = os.path.join(self.image_root, fixed_path)
        if os.path.exists(full_path): return full_path

        # 3. 폴더 구조 직접 탐색 (images_meg/{category}/{filename})
        filename = os.path.basename(fixed_path)
        if '_' in filename:
            category = filename.rsplit('_', 1)[0]
            try_path = os.path.join(self.image_root, 'images_meg', category, filename)
            if os.path.exists(try_path): return try_path
            
        return None

    def __getitem__(self, idx):
        # 1. Dictionary에서 경로 찾기
        cat = self.category_nr[idx]
        ex = self.exemplar_nr[idx]
        
        rel_path = IMAGE_MAP.get((cat, ex), None)
        
        if rel_path is None:
            # 매핑 실패 시 검은 화면 (거의 발생 안 함)
            img = Image.new('RGB', (512, 512))
        else:
            # 기본 경로 + Path Recovery 시도
            full_path = self._get_full_path(rel_path)
            
            # _get_full_path가 실패했을 경우 (dictionary path 불일치 대비)
            if not full_path:
                fname = os.path.basename(rel_path)
                if '_' in fname:
                    # e.g., crayon_15s.jpg -> category: crayon
                    # e.g., air_conditioner_01b.jpg -> category: air_conditioner
                    
                    # Try splitting by last underscore first (common case)
                    cat_guess = fname.rsplit('_', 1)[0]
                    fallback_path = os.path.join(self.image_root, 'images_meg', cat_guess, fname)
                    if os.path.exists(fallback_path):
                        full_path = fallback_path
                    else:
                        # Try iteratively for cases with multiple underscores if needed
                        parts = fname.split('_')
                        for i in range(1, len(parts)):
                            sub_cat = "_".join(parts[:i])
                            try_path = os.path.join(self.image_root, 'images_meg', sub_cat, fname)
                            if os.path.exists(try_path):
                                full_path = try_path
                                break
            
            if full_path:
                try:
                    img = Image.open(full_path).convert('RGB')
                except:
                    img = Image.new('RGB', (512, 512))
            else:
                # Still not found
                img = Image.new('RGB', (512, 512))
        # 2. Transform masking
        tensor_img = self.transform(img)
        tensor_img = tensor_img * 2 - 1  # [0,1] -> [-1, 1]
        
        valid_flag = 1 if rel_path and full_path else 0
        
        return tensor_img, valid_flag

# ==========================================
# 4. 모델 로드 (AutoKL)
# ==========================================
print("🚀 Loading Versatile Diffusion Model (AutoKL)...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth' 

if not os.path.exists(pth):
    raise FileNotFoundError(f"Checkpoint not found at {pth}")

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# 최적화: 불필요한 모듈 제거
net.clip = None
net.model = None

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
net.autokl = net.autokl.to(device)
net.autokl.eval() 
net.autokl.half() # FP16

print(f"✅ AutoKL Model Ready on {device}")

# ==========================================
# 5. 추출 및 HDF5 저장 함수
# ==========================================
def extract_and_save_hdf5(dataset, save_path):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_samples = len(dataset)
    
    # HDF5 생성
    print(f"💾 Creating HDF5: {save_path}")
    print(f"   Target Shape: ({num_samples}, 4, 64, 64)")
    
    total_missing = 0

    with h5py.File(save_path, 'w') as f:
        # FP16으로 저장 (용량 50% 절약)
        dset = f.create_dataset('features', shape=(num_samples, 4, 64, 64), dtype='float16', chunks=True)
        
        start_idx = 0
        with torch.no_grad():
            for batch_imgs, batch_valid_flags in tqdm(loader, desc=f"Extracting"):
                batch_imgs = batch_imgs.to(device).half()
                
                # Count missing
                # batch_valid_flags is a tensor of 0s and 1s
                missing_in_batch = (1 - batch_valid_flags).sum().item()
                total_missing += missing_in_batch
                
                # AutoKL Encode
                # autokl_encode는 (Mean, Logvar)가 아니라 Sampled Latent를 반환함
                z = net.autokl_encode(batch_imgs) 
                
                # CPU로 이동 및 저장
                z_np = z.cpu().numpy().astype(np.float16)
                
                batch_len = z_np.shape[0]
                dset[start_idx : start_idx + batch_len] = z_np
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
train_out_path = os.path.join(args.out_dir, 'combined_autokl_train.h5')

if os.path.exists(train_h5):
    if not os.path.exists(train_out_path):
        print(f"\n🚀 Processing Combined Train Data...")
        train_dataset = HDF5ImageDataset(train_h5, args.image_root)
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
    
    # 저장 경로: ./data/extracted_features/P1/P1_autokl_test.h5
    sub_out_dir = os.path.join(args.out_dir, sub)
    os.makedirs(sub_out_dir, exist_ok=True)
    test_out_path = os.path.join(sub_out_dir, f'{sub}_autokl_test.h5')
    
    if os.path.exists(test_h5):
        if not os.path.exists(test_out_path):
            print(f"\n🚀 Processing Test Data for {sub}...")
            test_dataset = HDF5ImageDataset(test_h5, args.image_root)
            extract_and_save_hdf5(test_dataset, test_out_path)
        else:
            print(f"⚠ Skipping {sub}: {test_out_path} already exists.")
    else:
        print(f"🚨 Test HDF5 not found for {sub}: {test_h5}")

print("\n🎉 All Feature Extractions Finished!")