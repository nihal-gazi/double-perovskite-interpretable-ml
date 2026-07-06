import os
import pandas as pd
import numpy as np
from pathlib import Path
from mendeleev import element
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & PATHS
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
CACHE_PATH = BASE_DIR / "Data" / "master_chemistry_cache.csv"

# ==========================================
# 2. FEATURE ENGINEERING (With Caching)
# ==========================================
if CACHE_PATH.exists():
    print(f"⚡ Loading chemistry data instantly from cache: {CACHE_PATH}")
    df_clean = pd.read_csv(CACHE_PATH)
else:
    print("⏳ Cache not found. Querying Mendeleev database (this will take a few minutes)...")
    df = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')
    
    def get_chem_props(el_symbol):
        if pd.isna(el_symbol) or str(el_symbol).strip() == '':
            return np.nan, np.nan, np.nan
        try:
            el = element(str(el_symbol).strip())
            val = el.nvalence()
            en = el.electronegativity()
            # Safely average Shannon Ionic Radii, fallback to covalent
            if el.ionic_radii:
                ionic_r = np.mean([ir.ionic_radius for ir in el.ionic_radii if ir.ionic_radius is not None])
                if ionic_r > 10: ionic_r /= 100 
            else:
                ionic_r = el.covalent_radius_pyykko / 100 if el.covalent_radius_pyykko else np.nan
            return val, en, ionic_r
        except:
            return np.nan, np.nan, np.nan

    print("Fetching atomic properties...")
    df['Val_A'], df['EN_A'], df['Shannon_A'] = zip(*df['element1'].apply(get_chem_props))
    df['Val_B'], df['EN_B'], df['Shannon_B'] = zip(*df['element3'].apply(get_chem_props))
    df['Val_Bprime'], df['EN_Bprime'], df['Shannon_Bprime'] = zip(*df['element4'].apply(get_chem_props))
    
    O_SHANNON = 1.40 
    eps = 1e-9
    
    df['d_AO'] = df['Shannon_A'] + O_SHANNON
    df['d_BO'] = df['Shannon_B'] + O_SHANNON
    df['d_BprimeO'] = df['Shannon_Bprime'] + O_SHANNON
    
    # Advanced Macros
    df['d_B_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
    df['Tolerance_Factor'] = df['d_AO'] / (np.sqrt(2) * df['d_B_avg'] + eps)
    df['Octahedral_Mismatch'] = np.abs(df['d_BO'] - df['d_BprimeO']) / (df['d_B_avg'] + eps)
    df['EN_avg'] = (df['EN_B'] + df['EN_Bprime']) / 2
    df['EN_diff'] = np.abs(df['EN_B'] - df['EN_Bprime'])
    df['Val_avg'] = (df['Val_B'] + df['Val_Bprime']) / 2

    # Save to cache keeping ALL targets
    keep_cols = [
        'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
        'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
        'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'EN_diff', 'Val_avg',
        'Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 
        'Band Gap (eV)', 'Total Magnetization (Î¼B)'
    ]
    df_clean = df[keep_cols].copy()
    
    # Convert to numeric
    for col in keep_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    df_clean.to_csv(CACHE_PATH, index=False)
    print(f"✅ Master dataset cached successfully to: {CACHE_PATH}")

# ==========================================
# 3. GB BENCHMARKING FOR ALL TARGETS
# ==========================================
features = [
    'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
    'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'EN_diff', 'Val_avg'
]

targets = [
    'Band Gap (eV)', 
    'Total Magnetization (Î¼B)', 
    'Energy Above Hull (eV)'
]

print(f"\n{'='*60}\nGRADIENT BOOSTING: FEATURE DISCOVERY\n{'='*60}")

for target in targets:
    print(f"\n🎯 TARGET: {target}")
    
    # Drop rows where target is NaN, and fill feature NaNs with 0
    d_target = df_clean.dropna(subset=[target]).copy()
    d_target[features] = d_target[features].fillna(0)
    
    X = d_target[features]
    y = d_target[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   📊 R² Score : {r2:.3f}")
    print(f"   📉 MAE      : {mae:.3f}")
    
    # Extract Top 5 Features
    importances = gb.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(5)
    
    print("   🏆 Top 5 Drivers:")
    for _, row in fi_df.iterrows():
        print(f"      - {row['Feature']:<20}: {row['Importance']:.1%}")