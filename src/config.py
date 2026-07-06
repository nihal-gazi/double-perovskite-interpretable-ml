import os
from pathlib import Path

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CIF_DIR = DATA_DIR / "CIF_Files"
MODELS_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"

# Ensure output directories exist
CIF_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Path to cached datasets
CHGNET_CACHE = PROCESSED_DATA_DIR / "chgnet_quantum_cache.csv"
THREE_D_CACHE = PROCESSED_DATA_DIR / "3D_chemistry_cache.csv"
CHEMISTRY_CACHE = PROCESSED_DATA_DIR / "master_chemistry_cache.csv"
ORIGINAL_DATA = RAW_DATA_DIR / "true and non true double perovskite sort.xlsx"

API_KEY = "gWJXczH9PXlsJ4tByN7ilvwJGv0TMnsY"

# ==========================================
# 2. HARDCODED PHYSICS DICTIONARIES
# ==========================================
A_OXIDATION_DICT = {
    'Li':1, 'Na':1, 'K':1, 'Rb':1, 'Cs':1,
    'Mg':2, 'Ca':2, 'Sr':2, 'Ba':2, 'Pb':2, 'Cd':2,
    'Sc':3, 'Y':3, 'La':3, 'Bi':3, 'Nd':3, 'Sm':3, 'Eu':3, 'Gd':3, 'Dy':3, 'Yb':3, 'Pr':3, 'Ce':3
}

TM_VALENCE_DICT = {
    'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':8, 'Co':9, 'Ni':10, 'Cu':11, 'Zn':12,
    'Zr':4, 'Nb':5, 'Mo':6, 'Tc':7, 'Ru':8, 'Rh':9, 'Pd':10, 'Ag':11, 'Cd':12,
    'Hf':4, 'Ta':5, 'W':6, 'Re':7, 'Os':8, 'Ir':9, 'Pt':10, 'Au':11, 'Hg':12
}

EN_DICT = {
    'H': 2.2, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.9, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'K': 0.82, 'Ca': 1.0, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
    'Co': 1.88, 'Ni': 1.91, 'Cu': 1.9, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2,
    'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17, 'Eu': 1.2, 'Gd': 1.2,
    'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27,
    'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54,
    'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02
}
