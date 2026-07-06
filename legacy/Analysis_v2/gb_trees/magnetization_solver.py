import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & HARDCODED PHYSICS DICTIONARIES
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "master_chemistry_cache.csv" # We use your cached data!

# Typical Oxidation States for A-site atoms (Alkali = +1, Alkaline Earth = +2, Rare Earth = +3)
A_OXIDATION_DICT = {
    'Li':1, 'Na':1, 'K':1, 'Rb':1, 'Cs':1,
    'Mg':2, 'Ca':2, 'Sr':2, 'Ba':2, 'Pb':2, 'Cd':2,
    'Sc':3, 'Y':3, 'La':3, 'Bi':3, 'Nd':3, 'Sm':3, 'Eu':3, 'Gd':3, 'Dy':3, 'Yb':3, 'Pr':3, 'Ce':3
}

# Total neutral valence electrons for Transition Metals (Group Number)
TM_VALENCE_DICT = {
    'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':8, 'Co':9, 'Ni':10, 'Cu':11, 'Zn':12,
    'Zr':4, 'Nb':5, 'Mo':6, 'Tc':7, 'Ru':8, 'Rh':9, 'Pd':10, 'Ag':11, 'Cd':12,
    'Hf':4, 'Ta':5, 'W':6, 'Re':7, 'Os':8, 'Ir':9, 'Pt':10, 'Au':11, 'Hg':12
}

def get_a_ox(el):
    return A_OXIDATION_DICT.get(str(el).strip(), np.nan)

def get_tm_val(el):
    return TM_VALENCE_DICT.get(str(el).strip(), np.nan)

# ==========================================
# 2. FEATURE ENGINEERING (QUANTUM MECHANICS)
# ==========================================
print("⏳ Loading cached data and engineering quantum features...")
# We need the original Excel file just to get the element strings again easily
ORIGINAL_DATA = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
df_orig = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')

# Merge the elements back into your cached dataframe
df = pd.read_csv(DATA_PATH)
df['element1'] = df_orig['element1']
df['element3'] = df_orig['element3']
df['element4'] = df_orig['element4']

# 1. Calculate the Charge supplied by A-sites
df['A_Oxidation'] = df['element1'].apply(get_a_ox)
df['Total_A_Charge'] = 2 * df['A_Oxidation']

# 2. Calculate the Charge REQUIRED from B and B' sites
df['Charge_Required_B_Bprime'] = 12 - df['Total_A_Charge']

# 3. Get neutral valence electrons of B and B'
df['Neutral_Val_B'] = df['element3'].apply(get_tm_val)
df['Neutral_Val_Bprime'] = df['element4'].apply(get_tm_val)

# 4. Calculate exactly how many d-electrons are left in the lattice!
df['Total_d_electrons'] = (df['Neutral_Val_B'] + df['Neutral_Val_Bprime']) - df['Charge_Required_B_Bprime']

# 5. Spin Proxy (Distance from half-filled shell. 5 d-electrons = Max Spin)
# If a system has 6 d-electrons, it has fewer unpaired spins than a 5 d-electron system.
df['Spin_Proxy_Distance'] = np.abs(df['Total_d_electrons'] - 5)

O_SHANNON = 1.40
df['d_BO'] = df['Shannon_B'] + O_SHANNON
df['d_BprimeO'] = df['Shannon_Bprime'] + O_SHANNON
df['d_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2

# ==========================================
# 3. GRADIENT BOOSTING FOR MAGNETIZATION
# ==========================================
# We now include the new Quantum Features alongside the structural ones
quantum_features = [
    'Total_d_electrons', 'Spin_Proxy_Distance', 'Neutral_Val_B', 'Neutral_Val_Bprime',
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'd_avg'
]

target = 'Total Magnetization (Î¼B)'

# Clean NaNs 
d_target = df.dropna(subset=[target] + quantum_features).copy()

X = d_target[quantum_features]
y = d_target[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n{'='*60}\nSOLVING TOTAL MAGNETIZATION\n{'='*60}")
gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 R² Score : {r2:.3f}")
print(f"📉 MAE      : {mae:.3f} Î¼B\n")

print("🏆 Top Drivers for Magnetization:")
importances = gb.feature_importances_
fi_df = pd.DataFrame({'Feature': quantum_features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=False)

for _, row in fi_df.iterrows():
    print(f"   - {row['Feature']:<20}: {row['Importance']:.1%}")