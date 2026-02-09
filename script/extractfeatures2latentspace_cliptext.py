import sys
import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
parser.add_argument("--out_dir", default='./data/extracted_features', help="Output directory")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", default='cuda', help="Device to use")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ==========================================
# 2. 이미지 경로 복원용 Dictionary 로드
# ==========================================
DICT_PATH = os.path.join(args.data_dir, 'image_path_dictionary.h5')
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"🚨 Dictionary file not found: {DICT_PATH}\nRun 'preprocess_optimized.py' first.")

print(f"📚 Loading Dictionary from {DICT_PATH}...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
    
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p

print(f"✅ Dictionary Loaded. {len(IMAGE_MAP)} unique images mapped.")

# ==========================================
# 3. Text Dataset 정의
# ==========================================
class TextPromptDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        
        # 메타데이터 로드
        with h5py.File(h5_path, 'r') as f:
            self.category_nr = f['category_nr'][:]
            self.exemplar_nr = f['exemplar_nr'][:]
            self.length = len(self.category_nr)

    def __len__(self):
        return self.length
    
    def _extract_category_name(self, path_str):
        """
        이미지 경로에서 카테고리 이름 추출
        예: 'images_meg/dog/dog_01.jpg' -> 'dog'
        """
        if not path_str: return "object"
        
        filename = os.path.basename(path_str)
        parent_dir = os.path.basename(os.path.dirname(path_str))
        
        category = ""
        
        # 1. 폴더명 사용
        if "images" not in parent_dir and parent_dir != ".":
            category = parent_dir
        else:
            # 2. 파일명에서 유추
            if '_' in filename:
                category = filename.rsplit('_', 1)[0]
            else:
                category = os.path.splitext(filename)[0]
                
        # 텍스트 정제
        category = category.replace('_', ' ')
        return category

    def __getitem__(self, idx):
        cat = self.category_nr[idx]
        ex = self.exemplar_nr[idx]
        
        # 경로 복원 -> 카테고리명 추출
        path_str = IMAGE_MAP.get((cat, ex), "")
        category_name = self._extract_category_name(path_str)
        
        # 프롬프트 생성 (카테고리명 그대로 사용)
        prompt = category_name 
        
        return prompt

# ==========================================
# 4. 모델 로드 (CLIP Text Encoder - FP32)
# ==========================================
print("🚀 Loading Versatile Diffusion Model (CLIP Text)...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth' 

if not os.path.exists(pth):
    raise FileNotFoundError(f"Checkpoint not found at {pth}")

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# 최적화: 불필요한 모듈 제거
net.autokl = None
net.model = None

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# [핵심 변경] FP32(float)로 설정
net.clip = net.clip.to(device).float()
net.clip.eval() 

print(f"✅ CLIP Text Model Ready on {device} (FP32)")

# ==========================================
# 5. 추출 및 HDF5 저장 함수 (FP32)
# ==========================================
def extract_and_save_hdf5(dataset, save_path):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_samples = len(dataset)
    
    # Feature Dimensions: [Batch, 77, 768]
    n_tokens = 77
    n_dim = 768
    
    print(f"💾 Creating Text Feature HDF5 (FP32): {save_path}")
    print(f"   Target Shape: ({num_samples}, {n_tokens}, {n_dim})")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # [변경] 저장 타입 float32
        dset = f.create_dataset('features', shape=(num_samples, n_tokens, n_dim), dtype='float32', chunks=True)
        
        start_idx = 0
        with torch.no_grad():
            for batch_prompts in tqdm(loader, desc=f"Extracting"):
                # batch_prompts는 문자열 리스트이므로 .to() 불필요
                
                # CLIP Encode Text (모델이 FP32이므로 출력도 FP32 Tensor)
                c = net.clip_encode_text(batch_prompts) 
                
                # [변경] numpy 변환 시 float32 유지
                c_np = c.cpu().numpy().astype(np.float32)
                
                batch_len = c_np.shape[0]
                dset[start_idx : start_idx + batch_len] = c_np
                start_idx += batch_len
    
    print(f"✅ Saved to {save_path}")

# ==========================================
# 6. 메인 실행 (Train & Test)
# ==========================================

# ------------------------------------------------
# 6-1. Combined Train Feature Extraction
# ------------------------------------------------
train_h5 = os.path.join(args.data_dir, 'train', 'combined_train.h5')
train_out_path = os.path.join(args.out_dir, 'combined_clip_text_train.h5')

if os.path.exists(train_h5):
    if not os.path.exists(train_out_path):
        print(f"\n🚀 Processing Combined Train Data (Text)...")
        train_dataset = TextPromptDataset(train_h5)
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
    sub_out_dir = os.path.join(args.out_dir, sub)
    test_out_path = os.path.join(sub_out_dir, f'{sub}_clip_text_test.h5')
    
    if os.path.exists(test_h5):
        if not os.path.exists(test_out_path):
            print(f"\n🚀 Processing Test Data for {sub} (Text)...")
            test_dataset = TextPromptDataset(test_h5)
            extract_and_save_hdf5(test_dataset, test_out_path)
        else:
            print(f"⚠ Skipping {sub}: {test_out_path} already exists.")
    else:
        print(f"🚨 Test HDF5 not found for {sub}: {test_h5}")

print("\n🎉 All CLIP Text Extractions Finished!")