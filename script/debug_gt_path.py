import h5py
import os
import argparse

DATA_DIR = './data'
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', 'P1_test.h5')

print(f"📚 Loading Image Dictionary...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p

print("📂 Loading Test Metadata...")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

indices = [0, 1]
for idx in indices:
    cat_id = test_cats[idx]
    ex_id = test_exs[idx]
    print(f"\nIndex {idx}: Cat {cat_id}, Ex {ex_id}")

    rel_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
    print(f"Rel Path: '{rel_path}'")

    full_path = os.path.join(DATA_DIR, rel_path)
    print(f"Full Path: '{full_path}'")

    exists = os.path.exists(full_path)
    print(f"Exists? {exists}")

if not exists:
    print("List dir of expected parent:")
    parent = os.path.dirname(full_path)
    if os.path.exists(parent):
        print(os.listdir(parent))
    else:
        print(f"Parent {parent} does not exist.")
        
    # Check if maybe it shouldn't have ./data prefix?
    if os.path.exists(rel_path):
        print(f"Wait, {rel_path} exists directly!")
