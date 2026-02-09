import torch
import os

def check_ckpt(path):
    print(f"Checking {path}...")
    try:
        sd = torch.load(path, map_location='cpu')
        
        # If it's a full model checkopint (optimizer, etc), get model state dict
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
            
        has_nan = False
        for k, v in sd.items():
            if torch.isnan(v).any():
                print(f"🚨 NaN detected in {k}")
                has_nan = True
                break # Stop after first nan
        
        if not has_nan:
            print("✅ No NaNs detected in checkpoint parameters.")
        else:
            print("❌ Checkpoint contains NaNs!")
            
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    import glob
    files = sorted(glob.glob('./checkpoints/autokl_final/*.pth'))
    for f in files:
        check_ckpt(f)
