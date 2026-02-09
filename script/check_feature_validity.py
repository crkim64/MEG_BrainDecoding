import h5py
import numpy as np
import os
import sys

def check_h5(path, name):
    print(f"\n🔍 Checking {name}: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found!")
        return
    
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print(f"   Keys: {keys}")
        if 'features' not in keys:
            print("❌ 'features' key missing")
            return
            
        # Original check on first 10 elements
        data_first_10 = f['features'][:10]
        print(f"   Shape: {f['features'].shape}")
        print(f"   Dtype: {data_first_10.dtype}")
        
        # Stats for first 10 elements
        d_min = data_first_10.min()
        d_max = data_first_10.max()
        d_mean = data_first_10.mean()
        d_std = data_first_10.std()
        
        print(f"   Min: {d_min:.4f}, Max: {d_max:.4f}, Mean: {d_mean:.4f}, Std: {d_std:.4f}")
        
        # New comprehensive checks (loading all data)
        data = f['features'][:] # Load all to memory if possible (for P1 it fits: 150MB~3GB)
        
        # Explicit NaN/Inf check (cast to float32 first to avoid overflow in stats)
        data_f32 = data.astype(np.float32)
        
        has_nan = np.isnan(data_f32).any()
        has_inf = np.isinf(data_f32).any()
        
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        
        # Check specific indices (0 and 1)
        if data_f32.shape[0] > 0:
            print(f"   Index 0 Stats: Min={data_f32[0].min():.4f}, Max={data_f32[0].max():.4f}, Mean={data_f32[0].mean():.4f}")
        if data_f32.shape[0] > 1:
            print(f"   Index 1 Stats: Min={data_f32[1].min():.4f}, Max={data_f32[1].max():.4f}, Mean={data_f32[1].mean():.4f}")

        print(f"   Global Stats: Min={data_f32.min():.4f}, Max={data_f32.max():.4f}, Mean={data_f32.mean():.4f}, Std={data_f32.std():.4f}")
        
        if d_max == 0 and d_min == 0:
            print("❌ WARNING: Data is all zeros!")
        elif d_std < 1e-6:
             print("❌ WARNING: Data has near-zero variance!")
        else:
            print("✅ Data looks valid (statistically).")

def main():
    root = './data/extracted_features/P1'
    files = [
        ('P1_autokl_test.h5', 'AutoKL'),
        ('P1_clip_vision_test.h5', 'CLIP Vision'),
        ('P1_clip_text_test.h5', 'CLIP Text')
    ]
    
    for fname, label in files:
        check_h5(os.path.join(root, fname), label)

if __name__ == "__main__":
    main()
