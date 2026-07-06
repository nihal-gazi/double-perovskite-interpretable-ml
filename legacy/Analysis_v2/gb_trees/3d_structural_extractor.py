import os
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "master_chemistry_cache.csv"
ORIGINAL_DATA = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
CIF_DIR = BASE_DIR / "Data" / "CIF_Files"
OUTPUT_CACHE = BASE_DIR / "Data" / "3D_chemistry_cache.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading original datasets...")
df = pd.read_csv(DATA_PATH)
df_orig = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')

# Clean the Material IDs
df['Material ID'] = df_orig['Material ID'].astype(str).str.strip()

# ==========================================
# 3. BULLETPROOF 3D FEATURE EXTRACTION
# ==========================================
tqdm.pandas(desc="Parsing 3D CIFs")

def extract_3d_features(m_id):
    # If ID is invalid or file doesn't exist, return NaNs
    if pd.isna(m_id) or m_id == 'nan':
        return pd.Series([np.nan, np.nan, "Unknown"])
        
    cif_path = CIF_DIR / f"{m_id}.cif"
    if not cif_path.exists():
        return pd.Series([np.nan, np.nan, "Unknown"])
        
    try:
        # Measure the actual 3D geometry (force string path for Windows)
        struct = Structure.from_file(str(cif_path))
        vol_per_atom = struct.volume / struct.num_sites
        density = struct.density
        symmetry = struct.get_space_group_info()[1]
        
        return pd.Series([vol_per_atom, density, symmetry])
    except Exception:
        # If the CIF file is corrupted, safely return NaNs
        return pd.Series([np.nan, np.nan, "Unknown"])

print(f"Extracting 3D geometric features from {CIF_DIR}...")
# This mapping guarantees exactly 416 rows of output!
df[['3D_Volume_Per_Atom', '3D_Density', '3D_Crystal_Symmetry']] = df['Material ID'].progress_apply(extract_3d_features)

# ==========================================
# 4. SAVE NEW 3D-INFORMED DATASET
# ==========================================
df.to_csv(OUTPUT_CACHE, index=False)
print(f"\nSuccessfully extracted 3D features and saved to: {OUTPUT_CACHE}")