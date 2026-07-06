import os
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure
from chgnet.model.model import CHGNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
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
OUTPUT_CACHE = BASE_DIR / "Data" / "chgnet_quantum_cache.csv"

print("Loading dataset...")
if OUTPUT_CACHE.exists():
    print(f"Loading data from cache: {OUTPUT_CACHE}")
    df = pd.read_csv(OUTPUT_CACHE)
else:
    df = pd.read_csv(DATA_PATH)

df_orig = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')
df['Material ID'] = df_orig['Material ID'].astype(str).str.strip()

# Run the inference if not already cached
if 'CHGNet_Energy' not in df.columns or df['CHGNet_Energy'].isnull().all():
    # ==========================================
    # 2. INITIALIZE THE ORACLE (CHGNet)
    # ==========================================
    print("Loading pre-trained CHGNet (Berkeley Materials Project)...")
    chgnet = CHGNet.load()

    # ==========================================
    # 3. GRAPH NEURAL NETWORK FEATURE EXTRACTION
    # ==========================================
    print(f"Passing 3D crystals through the Quantum Oracle...")
    tqdm.pandas(desc="CHGNet Inference")

    def get_chgnet_features(m_id):
        if pd.isna(m_id) or not str(m_id).startswith('mp-'):
            return pd.Series([np.nan, np.nan, np.nan])

        cif_path = CIF_DIR / f"{m_id}.cif"
        if not cif_path.exists():
            return pd.Series([np.nan, np.nan, np.nan])

        try:
            struct = Structure.from_file(str(cif_path))
            
            # Ask the Neural Network to solve the Schrödinger equation for this crystal
            prediction = chgnet.predict_structure(struct)
            
            # Extract Deep Quantum Features
            energy_per_atom = prediction['e'] / len(struct)
            
            # CHGNet predicts the magnetic spin of EVERY individual atom
            magmoms = prediction['m']
            net_magnetization = np.abs(np.sum(magmoms))       # Net magnetic moment
            abs_magnetization = np.sum(np.abs(magmoms))       # Total unpaired spins (handles antiferromagnetism)
            
            return pd.Series([energy_per_atom, net_magnetization, abs_magnetization])
        except Exception:
            return pd.Series([np.nan, np.nan, np.nan])

    df[['CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom']] = df['Material ID'].progress_apply(get_chgnet_features)

    # Save to cache so you never have to wait for CHGNet again!
    df.to_csv(OUTPUT_CACHE, index=False)
    print(f"Deep Quantum Features saved to: {OUTPUT_CACHE}")
else:
    print("Loaded cached CHGNet features instantly.")

# ==========================================
# 4. THE FINAL GRADIENT BOOSTING BENCHMARK
# ==========================================
print("\n" + "="*60)
print("FINAL TEST: GNN FEATURES + GB TREES")
print("="*60)

# We combine our best 1D formulas with our new Deep GNN features
features = [
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 
    'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
]

targets = ['Total Magnetization (Î¼B)', 'Band Gap (eV)', 'Energy Above Hull (eV)']

for target in targets:
    print(f"\nTARGET: {target}")
    
    # Clean NaNs
    d_target = df.dropna(subset=[target, 'CHGNet_Energy']).copy()
    d_target[features] = d_target[features].fillna(0)
    
    if len(d_target) < 50:
        continue
        
    X = d_target[features]
    y = d_target[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   R^2 Score : {r2:.4f}")
    print(f"   MAE       : {mae:.4f}")
    
    print("\n   Top Drivers:")
    fi_df = pd.DataFrame({'Feature': features, 'Importance': gb.feature_importances_})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(4)
    for _, row in fi_df.iterrows():
        print(f"      - {row['Feature']:<20}: {row['Importance']:.1%}")