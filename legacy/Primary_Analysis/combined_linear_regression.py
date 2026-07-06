import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
CHGNET_CACHE = BASE_DIR / "Data" / "chgnet_quantum_cache.csv"
THREE_D_CACHE = BASE_DIR / "Data" / "3D_chemistry_cache.csv"

if not CHGNET_CACHE.exists():
    raise FileNotFoundError(f"Missing CHGNet cache at: {CHGNET_CACHE}")
if not THREE_D_CACHE.exists():
    raise FileNotFoundError(f"Missing 3D cache at: {THREE_D_CACHE}")

# ==========================================
# 2. LOAD & MERGE DATA
# ==========================================
print("Loading datasets...")
df_chgnet = pd.read_csv(CHGNET_CACHE)
df_3d = pd.read_csv(THREE_D_CACHE)

# Clean and rename target columns to match standard formats
rename_dict = {}
for col in df_chgnet.columns:
    if 'Total Magnetization' in col:
        rename_dict[col] = 'Total Magnetization (uB)'
    elif 'Band Gap' in col:
        rename_dict[col] = 'Band Gap (eV)'
    elif 'Energy Above Hull' in col:
        rename_dict[col] = 'Energy Above Hull (eV)'
    elif 'Formation Energy' in col:
        rename_dict[col] = 'Formation Energy (eV/atom)'

df_chgnet = df_chgnet.rename(columns=rename_dict)
df_3d = df_3d.rename(columns=rename_dict)

# Merge datasets on Material ID
df_merged = pd.merge(df_chgnet, df_3d[['Material ID', '3D_Volume_Per_Atom', '3D_Density', '3D_Crystal_Symmetry']], on='Material ID', how='inner')
print(f"Successfully merged data. Total samples: {len(df_merged)}")

# Define all potential features
features = [
    # 1D chemical & atomic features
    'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
    'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
    'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'EN_diff', 'Val_avg',
    # 3D spatial properties
    '3D_Volume_Per_Atom', '3D_Density',
    # Deep GNN features
    'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
]

targets = [
    'Formation Energy (eV/atom)',
    'Total Magnetization (uB)',
    'Band Gap (eV)',
    'Energy Above Hull (eV)'
]

# Ensure 3D crystal symmetry is handled (force numeric space group or drop)
df_merged['3D_Crystal_Symmetry'] = pd.to_numeric(df_merged['3D_Crystal_Symmetry'], errors='coerce').fillna(0)

# ==========================================
# 3. LINEAR REGRESSION OPTIMIZER
# ==========================================
print("\n" + "="*80)
print("OPTIMIZING COMBINED LINEAR REGRESSION MODELS")
print("="*80)

results_summary = []

for target in targets:
    print(f"\nTarget Property: {target}")
    
    # Drop rows that have NaN in either the target or the feature columns
    clean_cols = features + [target]
    df_clean = df_merged.dropna(subset=clean_cols).copy()
    
    if len(df_clean) < 50:
        print(f"   Not enough data points ({len(df_clean)}). Skipping target.")
        continue
        
    X = df_clean[features]
    y = df_clean[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit various Linear models to find the best generalizable one
    models = {
        'OLS Linear Regression': LinearRegression(),
        'Ridge Regression (CV)': RidgeCV(alphas=np.logspace(-3, 3, 100)),
        'Lasso Regression (CV)': LassoCV(alphas=np.logspace(-3, 3, 100), max_iter=10000)
    }
    
    best_model_name = None
    best_model = None
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_y_pred = None
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   Model: {name:<25} | Test R^2 = {r2:.4f} | Test MAE = {mae:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_mae = mae
            best_model_name = name
            best_model = model
            best_y_pred = y_pred
            
    print(f"   Best Model: {best_model_name} (R^2 = {best_r2:.4f})")
    
    # Extract coefficients of best model
    coefs = best_model.coef_
    intercept = best_model.intercept_
    
    # Map coefficients to unscaled features (re-scaled for human interpretability)
    means = scaler.mean_
    stds = np.sqrt(scaler.var_)
    
    unscaled_coefs = coefs / stds
    unscaled_intercept = intercept - np.sum(coefs * means / stds)
    
    # Sort features by significance (magnitude of scaled coefficient)
    sig_df = pd.DataFrame({
        'Feature': features,
        'Scaled_Coefficient': coefs,
        'Unscaled_Coefficient': unscaled_coefs,
        'Magnitude': np.abs(coefs)
    }).sort_values('Magnitude', ascending=False)
    
    print("\n   Top 5 Significant Features:")
    for idx, row in sig_df.head(5).iterrows():
        print(f"      - {row['Feature']:<22}: Scaled Coef = {row['Scaled_Coefficient']:.4f} (Unscaled = {row['Unscaled_Coefficient']:.4e})")
        
    # Build readable equation using top 4 features and intercept
    equation_terms = []
    for _, row in sig_df.head(4).iterrows():
        sign = "+" if row['Unscaled_Coefficient'] >= 0 else "-"
        equation_terms.append(f"{sign} {abs(row['Unscaled_Coefficient']):.4f} * [{row['Feature']}]")
    
    sign_intercept = "+" if unscaled_intercept >= 0 else "-"
    equation_string = f"{target} = " + " ".join(equation_terms) + f" {sign_intercept} {abs(unscaled_intercept):.4f}"
    print(f"\n   Algebraic Equation Approximation (Top 4 Features):\n   {equation_string}")
    
    results_summary.append({
        'Target': target,
        'Best Model': best_model_name,
        'R2': best_r2,
        'MAE': best_mae,
        'Equation': equation_string,
        'Top Drivers': sig_df.head(5)['Feature'].tolist()
    })

# Write findings directly to file to copy over to Results.md later
print("\nLinear Regression analysis complete.")
