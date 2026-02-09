import sys
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Brain Module 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

CONFIG = {
    'batch_size': 1,
    'device': 'cuda',
    'ckpt_path': './checkpoints/clip_text/best_model.pth',
    'meg_path': './data/test/P1_test.h5',
    'feat_path': './checkpoints/clip_text/clip_text_stats.npz'
}

class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

def debug_prediction():
    # 1. Load Statistics
    print(f"Loading stats from {CONFIG['feat_path']}")
    stats = np.load(CONFIG['feat_path'])
    train_mean = torch.from_numpy(stats['mean']).float().to(CONFIG['device'])
    train_std = torch.from_numpy(stats['std']).float().to(CONFIG['device'])
    
    # 2. Load Model
    print("Loading Model...")
    model = BrainModule(
        in_channels={'meg': 271},
        out_dim_clip=768,
        out_dim_mse=77*768,
        time_len=281,
        hidden={'meg': 320},
        n_subjects=4, 
        merger=True, merger_pos_dim=512, merger_channels=270,
        rewrite=True, glu=1, glu_context=1,
        skip=True, batch_norm=True, post_skip=True, scale=1.0,
        subject_layers=True
    ).to(CONFIG['device'])

    if model.merger:
            model.merger.position_getter.get_positions = lambda batch: batch.meg_positions.to(CONFIG['device'])
            model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)
    
    # Load Weights
    sd = torch.load(CONFIG['ckpt_path'], map_location=CONFIG['device'])
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    # 3. Load Sample Data
    print("Loading MEG Data...")
    with h5py.File(CONFIG['meg_path'], 'r') as f:
        meg_data = f['meg'][:5] # Take 5 samples
    
    # Sensor Pos
    pos_path = './data/sensor_positions/sensor_positions_P1.npy'
    sensor_pos = np.load(pos_path)
    sensor_pos = torch.from_numpy(sensor_pos).float().unsqueeze(0).to(CONFIG['device'])
    subj_idx = torch.tensor([0]).long().to(CONFIG['device'])
    
    # 4. Predict & Compare
    with torch.no_grad():
        for i in range(5):
            print(f"\n--- Sample {i} ---")
            meg = torch.from_numpy(meg_data[i]).unsqueeze(0).float().to(CONFIG['device'])
            batch = BatchObject(meg, subj_idx, sensor_pos)
            
            # Predict
            out_clip, out_mse = model({'meg': meg}, batch) # out_mse: (1, 59136)
            
            raw_pred = out_mse.squeeze()
            
            # Stats
            p_min, p_max = raw_pred.min().item(), raw_pred.max().item()
            p_mean, p_std = raw_pred.mean().item(), raw_pred.std().item()
            
            t_min, t_max = train_mean.min().item(), train_mean.max().item()
            t_mean, t_std = train_mean.mean().item(), train_mean.std().item()
            
            print(f"Prediction | Min: {p_min:.4f}, Max: {p_max:.4f}, Mean: {p_mean:.4f}, Std: {p_std:.4f}")
            print(f"Train Stats| Min: {t_min:.4f}, Max: {t_max:.4f}, Mean: {t_mean:.4f}, Std: {t_std:.4f}")
            
            # Check for collapse
            if p_std < 1e-4:
                print("⚠️ WARNING: Prediction has near-zero variance (Collapse)")
            
            # Check value range consistency
            if abs(p_mean - t_mean) > 1.0:
                 print("⚠️ WARNING: Prediction mean shifted significantly from Train Mean")

            # Check normalization effect
            # Z-Score Pred
            z_pred = (raw_pred - p_mean) / (p_std + 1e-6)
            # Re-Norm
            renorm_pred = z_pred * train_std + train_mean
            
            rn_min, rn_max = renorm_pred.min().item(), renorm_pred.max().item()
            rn_mean, rn_std = renorm_pred.mean().item(), renorm_pred.std().item()
             
            print(f"Renormalized| Min: {rn_min:.4f}, Max: {rn_max:.4f}, Mean: {rn_mean:.4f}, Std: {rn_std:.4f}")

if __name__ == "__main__":
    debug_prediction()
