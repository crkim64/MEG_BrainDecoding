import h5py
import numpy as np

DICT_PATH = 'data/image_path_dictionary.h5'
TEST_PATH = 'data/test/P1_test.h5'

print(f"🔍 Loading Dictionary: {DICT_PATH}")
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    print(f"   Dictionary Keys Sample (First 5):")
    for i in range(5):
        print(f"     Key {i}: ({cats[i]}, {exs[i]}) - Types: {type(cats[i])}, {type(exs[i])}")
        
    # Build Map
    IMAGE_MAP = {}
    for c, e in zip(cats, exs):
        IMAGE_MAP[(c, e)] = True
    print(f"   Dictionary Size: {len(IMAGE_MAP)}")

print(f"\n🔍 Loading Test Data: {TEST_PATH}")
with h5py.File(TEST_PATH, 'r') as f:
    t_cats = f['category_nr'][:]
    t_exs = f['exemplar_nr'][:]
    print(f"   Test Keys Sample (First 5):")
    for i in range(5):
        print(f"     Key {i}: ({t_cats[i]}, {t_exs[i]}) - Types: {type(t_cats[i])}, {type(t_exs[i])}")
        
    # Check Lookups
    print(f"\n🔍 Checking Lookup for First 5 Test Items:")
    for i in range(5):
        key = (t_cats[i], t_exs[i])
        found = key in IMAGE_MAP
        print(f"     Lookup {key}: {'✅ Found' if found else '❌ FAILED'}")
        
        # Try type casting checking
        if not found:
            # Try float to int conversion if floats
            if isinstance(key[0], (float, np.floating)):
                k2 = (int(key[0]), int(key[1]))
                f2 = k2 in IMAGE_MAP
                print(f"       -> Trying int cast {k2}: {'✅ Found' if f2 else '❌ FAILED'}")

            # Try int to float
            if isinstance(key[0], (int, np.integer)):
                k3 = (float(key[0]), float(key[1]))
                f3 = k3 in IMAGE_MAP
                print(f"       -> Trying float cast {k3}: {'✅ Found' if f3 else '❌ FAILED'}")
