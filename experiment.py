import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import h5py

# 경로 설정 (사용자 환경에 맞게 수정)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 데이터셋 클래스 (train_brainmodule_autokl_final.py와 동일하게 구성)
class MEGAutoKLDatasetDebug(torch.utils.data.Dataset):
    def __init__(self, meg_path, autokl_path, subjects):
        self.meg_path = meg_path
        self.autokl_path = autokl_path
        self.subjects = subjects
        with h5py.File(meg_path, 'r') as f:
            self.length = f['meg'].shape[0]
            self.categories = f['category_nr'][:]
            self.exemplars = f['exemplar_nr'][:]
        self.meg_hf = None
        self.autokl_hf = None

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.meg_hf is None:
            self.meg_hf = h5py.File(self.meg_path, 'r')
            self.autokl_hf = h5py.File(self.autokl_path, 'r')
            
        meg_data = self.meg_hf['meg'][idx]
        
        # [핵심] AutoKL 읽기
        feat_raw = self.autokl_hf['features'][idx] 
        target_mse = torch.from_numpy(feat_raw.reshape(-1)).float()
        
        # CLIP Target (Mean)
        feat_mean = np.mean(feat_raw, axis=0).reshape(-1)
        target_clip = torch.from_numpy(feat_mean).float()
        
        # SoftCLIP ID
        cat_id = self.categories[idx]
        ex_id = self.exemplars[idx]
        unique_img_id = cat_id * 100 + ex_id
        
        return torch.from_numpy(meg_data), target_clip, target_mse, unique_img_id

def debug_main():
    print("🕵️ Debugging DataLoader...")
    data_dir = './data'
    train_meg = os.path.join(data_dir, 'train/combined_train.h5')
    train_autokl = os.path.join(data_dir, 'extracted_features/combined_autokl_train.h5')
    subjects = ['P1', 'P2', 'P3', 'P4']

    dataset = MEGAutoKLDatasetDebug(train_meg, train_autokl, subjects)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("✅ Dataset loaded. Fetching first batch...")
    
    try:
        meg, t_clip, t_mse, img_ids = next(iter(loader))
    except Exception as e:
        print(f"🚨 Error: {e}")
        return

    # 1. Target MSE 값 확인
    print(f"\n📊 [Target MSE Stats]")
    print(f"   Shape: {t_mse.shape}")
    print(f"   Min: {t_mse.min().item():.6f}")
    print(f"   Max: {t_mse.max().item():.6f}")
    print(f"   Mean: {t_mse.mean().item():.6f}")
    print(f"   Abs Mean: {t_mse.abs().mean().item():.6f}")
    
    if torch.allclose(t_mse, torch.zeros_like(t_mse), atol=1e-5):
        print("\n🚨🚨🚨 [CRITICAL] Target is ALL ZEROS! 로더 문제입니다.")
    else:
        print("\n✅ Target data looks valid (Not zero).")

    # 2. 중복 이미지 확인 (SoftCLIP 작동 여부)
    unique_ids, counts = torch.unique(img_ids, return_counts=True)
    num_duplicates = (counts > 1).sum().item()
    print(f"\n🧩 [Duplicate Check] Batch Size 128")
    print(f"   Duplicate Groups found: {num_duplicates}")
    if num_duplicates == 0:
        print("⚠️ [WARNING] 배치 내에 같은 이미지가 하나도 없습니다. SoftCLIP이 동작하지 않습니다.")

if __name__ == "__main__":
    debug_main()