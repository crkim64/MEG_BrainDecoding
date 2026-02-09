import numpy as np
import sys

file_path = "/home/kcr/BrainDiffuser_fMRI/brain-diffuser/data/extracted_features/subj01/nsd_vdvae_features_31l.npz"

try:
    data = np.load(file_path)
    print(f"File: {file_path}")
    print("Keys:", data.files)
    for k in data.files:
        print(f"Key: {k}, Shape: {data[k].shape}, Dtype: {data[k].dtype}")
except Exception as e:
    print(f"Error loading file: {e}")
