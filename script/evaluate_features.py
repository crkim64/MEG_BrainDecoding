
import sys
import os
import argparse
import numpy as np
import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add paths
sys.path.append('versatile_diffusion')
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PROJ_DIR))

try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

# ==========================================
# Configuration
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1')
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

DEVICE = args.device
print(f"🚀 Evaluating Feature Alignment for {args.subject} on {DEVICE}...")

# Paths
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
GT_VISION_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_clip_vision_test.h5')
GT_TEXT_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_clip_text_test.h5')

CKPT_VISION = './checkpoints/clip_vision/best_model.pth'
CKPT_TEXT = './checkpoints/clip_text/best_model.pth'

# ==========================================
# Helper Classes
# ==========================================
class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

def load_brain_module(ckpt_path, out_dim_clip, out_dim_mse, device):
    print(f"⌛ Loading BrainModule: {ckpt_path}")
    model = BrainModule(
        in_channels={'meg': 271},
        out_dim_clip=out_dim_clip,
        out_dim_mse=out_dim_mse,
        time_len=281,
        hidden={'meg': 320},
        n_subjects=4,
        merger=True, merger_pos_dim=512, merger_channels=270,
        rewrite=True, glu=1, glu_context=1,
        skip=True, batch_norm=True, post_skip=True, scale=1.0,
        subject_layers=True
    ).to(device)
    
    # Merger setup for pos
    if model.merger:
        model.merger.position_getter.get_positions = lambda batch: batch.meg_positions.to(device)
        model.merger.position_getter.is_invalid = lambda pos: torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool).to(pos.device)

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    else:
        print(f"⚠️ Checkpoint NOT FOUND: {ckpt_path}")
        return None

# ==========================================
# Main
# ==========================================
def main():
    # 1. Load Data
    print("📂 Loading Data...")
    
    # MEG
    with h5py.File(TEST_MEG_PATH, 'r') as f:
        meg_data = f['meg'][:]
    
    # GT Vision
    with h5py.File(GT_VISION_PATH, 'r') as f:
        gt_vision = f['features'][:] # (N, 257, 768)
        
    # GT Text
    with h5py.File(GT_TEXT_PATH, 'r') as f:
        gt_text = f['features'][:] # (N, 77, 768)

    print(f"   MEG Shape: {meg_data.shape}")
    print(f"   GT Vision Shape: {gt_vision.shape}")
    print(f"   GT Text Shape: {gt_text.shape}")

    # Prepare Sensor Pos
    pos_path = os.path.join(DATA_DIR, 'sensor_positions', f"sensor_positions_{args.subject}.npy")
    sensor_pos = np.load(pos_path) if os.path.exists(pos_path) else np.random.rand(271, 2)
    sensor_pos = torch.from_numpy(sensor_pos).float().unsqueeze(0).to(DEVICE)
    sub_idx = torch.tensor([{'P1':0, 'P2':1, 'P3':2, 'P4':3}[args.subject]]).long().to(DEVICE)

    # 2. Load Models
    bm_vision = load_brain_module(CKPT_VISION, 768, 257*768, DEVICE)
    bm_text = load_brain_module(CKPT_TEXT, 768, 77*768, DEVICE)

    # 3. Evaluate Loop
    dataset = TensorDataset(
        torch.from_numpy(meg_data).float(),
        torch.from_numpy(gt_vision).float(),
        torch.from_numpy(gt_text).float()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = {
        'vis_mse': [], 'vis_cos_full': [], 'vis_cos_global': [],
        'txt_mse': [], 'txt_cos_full': [], 'txt_cos_global': []
    }
    
    # Stats for Distribution Check
    stats = {
        'pred_vis_mean': [], 'pred_vis_std': [],
        'gt_vis_mean': [], 'gt_vis_std': [],
    }

    print("⚡ Running Inference & Evaluation...")
    with torch.no_grad():
        for meg, gt_vis, gt_txt in tqdm(loader):
            meg = meg.to(DEVICE)
            gt_vis = gt_vis.to(DEVICE)
            gt_txt = gt_txt.to(DEVICE)
            
            # Expand sensor_pos to match batch size
            batch_pos = sensor_pos.expand(meg.shape[0], -1, -1)
            batch = BatchObject(meg, sub_idx, batch_pos)

            # --- VISION ---
            if bm_vision:
                _, pred_vis_flat = bm_vision({'meg': meg}, batch)
                pred_vis = pred_vis_flat.view(-1, 257, 768)
                
                # MSE
                metrics['vis_mse'].append(F.mse_loss(pred_vis, gt_vis).item())
                
                # Cosine Full
                metrics['vis_cos_full'].append(F.cosine_similarity(pred_vis.flatten(1), gt_vis.flatten(1)).mean().item())
                
                # Cosine Global (Mean Pooling)
                metrics['vis_cos_global'].append(F.cosine_similarity(pred_vis.mean(1), gt_vis.mean(1)).mean().item())
                
                # Stats
                stats['pred_vis_mean'].append(pred_vis.mean().item())
                stats['pred_vis_std'].append(pred_vis.std().item())
                stats['gt_vis_mean'].append(gt_vis.mean().item())
                stats['gt_vis_std'].append(gt_vis.std().item())

            # --- TEXT ---
            if bm_text:
                _, pred_txt_flat = bm_text({'meg': meg}, batch)
                pred_txt = pred_txt_flat.view(-1, 77, 768)
                
                metrics['txt_mse'].append(F.mse_loss(pred_txt, gt_txt).item())
                metrics['txt_cos_full'].append(F.cosine_similarity(pred_txt.flatten(1), gt_txt.flatten(1)).mean().item())
                metrics['txt_cos_global'].append(F.cosine_similarity(pred_txt.mean(1), gt_txt.mean(1)).mean().item())

    # 4. Report
    print("\n📊 Evaluation Results:")
    print("="*40)
    
    if bm_vision:
        print(f"👁️ CLIP VISION:")
        print(f"   MSE: {np.mean(metrics['vis_mse']):.5f}")
        print(f"   Cosine Sim (Full Sequence):   {np.mean(metrics['vis_cos_full']):.4f}")
        print(f"   Cosine Sim (Global Average):  {np.mean(metrics['vis_cos_global']):.4f}")
        print("-" * 20)
        print(f"   Pred Stats | Mean: {np.mean(stats['pred_vis_mean']):.4f}, Std: {np.mean(stats['pred_vis_std']):.4f}")
        print(f"   GT   Stats | Mean: {np.mean(stats['gt_vis_mean']):.4f}, Std: {np.mean(stats['gt_vis_std']):.4f}")
        print("="*40)

    if bm_text:
        print(f"📝 CLIP TEXT:")
        print(f"   MSE: {np.mean(metrics['txt_mse']):.5f}")
        print(f"   Cosine Sim (Full Sequence):   {np.mean(metrics['txt_cos_full']):.4f}")
        print(f"   Cosine Sim (Global Average):  {np.mean(metrics['txt_cos_global']):.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
