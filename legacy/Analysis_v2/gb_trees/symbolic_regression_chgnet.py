import os
import pandas as pd
import numpy as np
from pathlib import Path
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils.validation import validate_data
import warnings

warnings.filterwarnings("ignore")

# --- scikit-learn compatibility ---
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "chgnet_quantum_cache.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cache file not found: {DATA_PATH}. Please run oracle_chgnet.py first.")

# ==========================================
# 2. LOAD DATA
# ==========================================
print(f"Loading cached dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

features = [
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 
    'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
]

targets = [
    'Total Magnetization (Î¼B)', 
    'Band Gap (eV)', 
    'Energy Above Hull (eV)'
]

# Ensure targets are read properly (handle characters)
rename_dict = {}
for col in df.columns:
    if 'Total Magnetization' in col:
        rename_dict[col] = 'Total Magnetization (uB)'
    elif 'Band Gap' in col:
        rename_dict[col] = 'Band Gap (eV)'
    elif 'Energy Above Hull' in col:
        rename_dict[col] = 'Energy Above Hull (eV)'

df = df.rename(columns=rename_dict)
targets_clean = ['Total Magnetization (uB)', 'Band Gap (eV)', 'Energy Above Hull (eV)']

print("\n" + "="*60)
print("RUNNING SYMBOLIC REGRESSION ON QUANTUM DESCRIPTORS")
print("="*60)

for target in targets_clean:
    print(f"\nTARGET: {target}")
    
    # Clean NaNs
    d_target = df.dropna(subset=[target, 'CHGNet_Energy']).copy()
    d_target[features] = d_target[features].fillna(0)
    
    if len(d_target) < 50:
        print("   Not enough data points. Skipping...")
        continue
        
    X = d_target[features]
    y = d_target[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    est_gp = SymbolicRegressor(
        population_size=3000,
        generations=50,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.002,
        random_state=42,
        function_set=('add', 'sub', 'mul', 'div', 'abs')
    )
    
    est_gp.fit(X_train, y_train)
    
    y_pred = est_gp.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   Test R^2 Score : {r2:.4f}")
    print(f"   Test MAE       : {mae:.4f}")
    
    # Format and display equation
    equation = str(est_gp._program)
    for i, feature in reversed(list(enumerate(features))):
        equation = equation.replace(f"X{i}", f"[{feature}]")
        
    print(f"   Discovered Equation:\n   {target} = {equation}")
