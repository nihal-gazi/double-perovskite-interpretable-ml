import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & HARDCODED PHYSICS
# ==========================================
API_KEY = "gWJXczH9PXlsJ4tByN7ilvwJGv0TMnsY"  

BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "master_chemistry_cache.csv"
ORIGINAL_DATA = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
CIF_DIR = BASE_DIR / "Data" / "CIF_Files"
CIF_DIR.mkdir(parents=True, exist_ok=True)

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

def get_a_ox(el): return A_OXIDATION_DICT.get(str(el).strip(), np.nan)
def get_tm_val(el): return TM_VALENCE_DICT.get(str(el).strip(), np.nan)

# ==========================================
# 2. LOAD DATA & QUANTUM MECHANICS
# ==========================================
print("⏳ Loading caches and engineering Quantum mechanics...")
df = pd.read_csv(DATA_PATH)
df_orig = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')

df['Material ID'] = df_orig['Material ID'].astype(str).str.strip()
df['element1'] = df_orig['element1']
df['element3'] = df_orig['element3']
df['element4'] = df_orig['element4']

df['Total_A_Charge'] = 2 * df['element1'].apply(get_a_ox)
df['Total_d_electrons'] = (df['element3'].apply(get_tm_val) + df['element4'].apply(get_tm_val)) - (12 - df['Total_A_Charge'])
df['Spin_Proxy_Distance'] = np.abs(df['Total_d_electrons'] - 5)

O_SHANNON = 1.40
df['d_BO'] = df['Shannon_B'] + O_SHANNON
df['d_BprimeO'] = df['Shannon_Bprime'] + O_SHANNON
df['d_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2

# ==========================================
# 3. BULLETPROOF 3D DOWNLOAD & PARSE
# ==========================================
print(f"\n🔬 Fetching and Parsing 3D Crystals...")
tqdm.pandas(desc="Processing CIFs")

def fetch_and_parse(m_id):
    if pd.isna(m_id) or not str(m_id).startswith('mp-'):
        return pd.Series([np.nan, np.nan, np.nan])

    cif_path = CIF_DIR / f"{m_id}.cif"
    struct = None

    # 1. Read file if it exists and isn't empty
    if cif_path.exists() and cif_path.stat().st_size > 100:
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception:
            pass

    # 2. If missing, fetch from correct API endpoint
    if struct is None:
        url = f"https://api.materialsproject.org/materials/summary/?material_ids={m_id}&_fields=structure"
        headers = {'X-API-KEY': API_KEY}
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data") and len(data["data"]) > 0:
                    struct_dict = data["data"][0].get("structure")
                    if struct_dict:
                        struct = Structure.from_dict(struct_dict)
                        struct.to(filename=str(cif_path), fmt="cif")
        except Exception:
            pass

    # 3. Measure physical properties (Removed volume_fraction)
    if struct is not None:
        try:
            vol = struct.volume / struct.num_sites
            den = struct.density
            sym = struct.get_space_group_info()[1]
            return pd.Series([vol, den, sym])
        except Exception:
            return pd.Series([np.nan, np.nan, np.nan])
    else:
        return pd.Series([np.nan, np.nan, np.nan])

df[['3D_Volume_Per_Atom', '3D_Density', '3D_Crystal_Symmetry']] = df['Material ID'].progress_apply(fetch_and_parse)

valid_3d = df['3D_Volume_Per_Atom'].notna().sum()
print(f"✅ Successfully loaded True 3D Physics for {valid_3d} materials!")

# ==========================================
# 4. GRADIENT BOOSTING: THE ULTIMATE TEST
# ==========================================
features = [
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'd_avg',
    'Total_d_electrons', 'Spin_Proxy_Distance',
    '3D_Volume_Per_Atom', '3D_Density', '3D_Crystal_Symmetry'
]

targets = ['Total Magnetization (Î¼B)', 'Band Gap (eV)']

for target in targets:
    print(f"\n{'='*50}\n🎯 TARGET: {target}\n{'='*50}")
    
    d_target = df.dropna(subset=[target, '3D_Volume_Per_Atom']).copy()
    d_target[features] = d_target[features].fillna(0)
    
    if len(d_target) < 50:
        print(f"⚠️ Only {len(d_target)} valid rows found. Skipping...")
        continue
        
    X = d_target[features]
    y = d_target[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingRegressor(n_estimators=350, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📊 R² Score : {r2:.3f}")
    print(f"📉 MAE      : {mae:.3f}")
    
    print("\n🏆 Top 5 Physical Drivers:")
    fi_df = pd.DataFrame({'Feature': features, 'Importance': gb.feature_importances_})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(5)
    
    for _, row in fi_df.iterrows():
        print(f"   - {row['Feature']:<20}: {row['Importance']:.1%}")