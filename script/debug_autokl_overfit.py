
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np

# Add path to model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.brainmodule import BrainModule

# Config
DEVICE = 'cuda:0'
BATCH_SIZE = 32
EPOCHS = 200 # More epochs to ensure convergence
LR = 1e-3 # Higher LR for overfitting

# Paths
MEG_PATH = './data/train/combined_train.h5'
AUTOKL_PATH = './data/extracted_features/combined_autokl_train.h5'

class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

def main():
    print(f"🚀 Debug: Overfitting Check on 1 Batch")
    
    # 1. Load 1 Batch of Data
    print("Loading data...")
    with h5py.File(MEG_PATH, 'r') as f_meg, h5py.File(AUTOKL_PATH, 'r') as f_kl:
        meg_data = f_meg['meg'][:BATCH_SIZE]
        subj_idx = f_meg['subject_idx'][:BATCH_SIZE]
        
        # AutoKL Features
        feat_raw = f_kl['features'][:BATCH_SIZE]
        targets = feat_raw.reshape(BATCH_SIZE, -1)
        
    # Fake Positions (Random is fine for overfitting check, or zeros)
    # Actually, BrainModule uses positions. Let's make them random but consistent.
    positions = torch.rand(BATCH_SIZE, 271, 2).to(DEVICE)
    
    meg_tensor = torch.from_numpy(meg_data).float().to(DEVICE)
    target_tensor = torch.from_numpy(targets).float().to(DEVICE)
    subj_tensor = torch.from_numpy(subj_idx).long().to(DEVICE)
    
    batch_obj = BatchObject(meg_tensor, subj_tensor, positions)
    
    # 2. Init Model
    print("Initializing BrainModule...")
    model = BrainModule(
        in_channels={'meg': 271},
        out_dim_clip=16384,
        out_dim_mse=16384,
        time_len=281,
        hidden={'meg': 320},
        n_subjects=4,
        merger=True, merger_pos_dim=512, merger_channels=270,
        rewrite=True, glu=1, glu_context=1,
        skip=True, batch_norm=True, post_skip=True, scale=1.0,
        subject_layers=True
    ).to(DEVICE)
    
    # Mock position getter
    if model.merger:
        model.merger.position_getter.get_positions = lambda b: b.meg_positions
        model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print("Starting Training Loop (MSE Only)...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        _, pred_mse = model({'meg': meg_tensor}, batch_obj)
        
        # Loss
        loss = criterion(pred_mse, target_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {loss.item():.8f}")
            
    print("Check finished.")

if __name__ == "__main__":
    main()
