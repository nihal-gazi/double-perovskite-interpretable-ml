import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
INPUT_XLSX = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Data\true and non true double perovskite sort.xlsx")

OUTPUT_DIR = Path.cwd() / "formation_energy_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"Excel file not found: {INPUT_XLSX}")

# =========================
# EN DICTIONARY
# =========================
EN_DICT = {
    'H': 2.2, 'He': np.nan, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
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

print("=" * 80)
print("OPTIMIZED FORMATION ENERGY PREDICTION")
print("=" * 80)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name='approx true double perovskite')
print(f"Loaded {len(df)} compounds\n")

# =========================
# FEATURE ENGINEERING
# =========================
print("Engineering features...")

def get_en(el):
    if pd.isna(el) or str(el).strip() == '':
        return np.nan
    return EN_DICT.get(str(el).strip(), np.nan)

# Base features
df['EN_A'] = df['element1'].apply(get_en)
df['EN_B'] = df['element3'].apply(get_en)
df['EN_Bprime'] = df['element4'].apply(get_en)
df['EN_O'] = 3.44

O_RADIUS = 1.40
df['d_AO'] = pd.to_numeric(df['metallic radius1'], errors='coerce') + O_RADIUS
df['d_BO'] = pd.to_numeric(df['metallic radius 3'], errors='coerce') + O_RADIUS
df['d_BprimeO'] = pd.to_numeric(df['metallic radius4'], errors='coerce') + O_RADIUS

eps = 1e-9

# Interaction features
df['EN_B_minus_A'] = df['EN_B'] - df['EN_A']
df['EN_Bprime_minus_A'] = df['EN_Bprime'] - df['EN_A']
df['EN_Bprime_minus_B'] = df['EN_Bprime'] - df['EN_B']
df['EN_ratio_Bprime_A'] = df['EN_Bprime'] / (df['EN_A'] + eps)
df['EN_ratio_B_A'] = df['EN_B'] / (df['EN_A'] + eps)

df['d_BprimeO_minus_BO'] = df['d_BprimeO'] - df['d_BO']
df['d_AO_minus_BO'] = df['d_AO'] - df['d_BO']
df['d_ratio_BprimeO_BO'] = df['d_BprimeO'] / (df['d_BO'] + eps)
df['d_ratio_AO_BO'] = df['d_AO'] / (df['d_BO'] + eps)

# Perovskite descriptors
df['d_B_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
df['tolerance_factor'] = df['d_AO'] / (np.sqrt(2) * df['d_B_avg'] + eps)
df['octahedral_factor'] = df['d_B_avg'] / (df['d_AO'] + eps)
df['distortion_idx'] = np.abs(df['d_BO'] - df['d_BprimeO']) / (df['d_B_avg'] + eps)

# Custom tolerance from symbolic regression
df['custom_tolerance'] = (df['EN_Bprime'] / (df['EN_A'] + eps)) / (df['d_BO'] / (df['d_BprimeO'] + eps) + eps)

# Combined features
df['EN_d_mismatch'] = df['EN_Bprime'] * df['d_BprimeO'] - df['EN_B'] * df['d_BO']
df['bond_strength_B'] = df['EN_B'] / (df['d_BO']**2 + eps)
df['bond_strength_Bprime'] = df['EN_Bprime'] / (df['d_BprimeO']**2 + eps)

# Additional summary features
df['EN_avg'] = (df['EN_B'] + df['EN_Bprime']) / 2
df['EN_std'] = np.abs(df['EN_B'] - df['EN_Bprime'])
df['d_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
df['d_std'] = np.abs(df['d_BO'] - df['d_BprimeO'])

all_features = [
    'EN_A', 'EN_B', 'EN_Bprime',
    'd_AO', 'd_BO', 'd_BprimeO',
    'EN_B_minus_A', 'EN_Bprime_minus_A', 'EN_Bprime_minus_B',
    'EN_ratio_Bprime_A', 'EN_ratio_B_A',
    'd_BprimeO_minus_BO', 'd_AO_minus_BO',
    'd_ratio_BprimeO_BO', 'd_ratio_AO_BO',
    'd_B_avg', 'tolerance_factor', 'octahedral_factor', 'distortion_idx',
    'custom_tolerance', 'EN_d_mismatch',
    'bond_strength_B', 'bond_strength_Bprime',
    'EN_avg', 'EN_std', 'd_avg', 'd_std'
]

target = 'Formation Energy (eV/atom)'
df[target] = pd.to_numeric(df[target], errors='coerce')

# Clean data
df_clean = df.dropna(subset=[target]).copy()
for col in all_features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_clean[all_features].copy()
y = df_clean[target].copy()

print(f"Features: {len(all_features)}, Samples: {len(X)}\n")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
print("=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {
    'Ridge (α=1)': Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ]),
    'Ridge (α=5)': Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=5.0))
    ]),
    'Ridge (α=10)': Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ]),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...", end=' ')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({'Model': name, 'R²': r2, 'MAE': mae})
    trained_models[name] = (model, y_pred)

    print(f"R² = {r2:.4f}, MAE = {mae:.4f}")

print()
results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print(results_df.to_string(index=False))

# Best model
best_name = results_df.iloc[0]['Model']
best_model, best_pred = trained_models[best_name]
best_r2 = float(results_df.iloc[0]['R²'])
best_mae = float(results_df.iloc[0]['MAE'])

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_name} (R² = {best_r2:.4f})")
print(f"{'='*80}\n")

# =========================
# FEATURE IMPORTANCE / COEFS
# =========================
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': all_features, 'Importance': imp}).sort_values('Importance', ascending=False)
    col_name = 'Importance'
else:
    ridge = best_model.named_steps['ridge']
    coef = np.abs(ridge.coef_)
    fi_df = pd.DataFrame({'Feature': all_features, 'Abs_Coef': coef, 'Coef': ridge.coef_}).sort_values('Abs_Coef', ascending=False)
    col_name = 'Abs_Coef'

print("Top 15 Features:")
print(fi_df.head(15).to_string(index=False))

# =========================
# VISUALIZATION
# =========================
print(f"\n{'='*80}")
print("CREATING VISUALIZATION")
print(f"{'='*80}\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, best_pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Formation Energy (eV/atom)', fontsize=11)
ax1.set_ylabel('Predicted Formation Energy (eV/atom)', fontsize=11)
ax1.set_title(f'{best_name}: Actual vs Predicted (R² = {best_r2:.4f})', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Residuals
ax2 = axes[0, 1]
residuals = y_test - best_pred
ax2.scatter(best_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Formation Energy (eV/atom)', fontsize=11)
ax2.set_ylabel('Residuals (eV/atom)', fontsize=11)
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Feature importance
ax3 = axes[1, 0]
top_n = 12
top_fi = fi_df.head(top_n)
y_pos = np.arange(len(top_fi))
ax3.barh(y_pos, top_fi[col_name], edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_fi['Feature'], fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel(col_name, fontsize=11)
ax3.set_title('Top 12 Important Features', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='x')

# Error distribution
ax4 = axes[1, 1]
ax4.hist(residuals, bins=25, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
ax4.set_xlabel('Residuals (eV/atom)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'Error Distribution (MAE = {best_mae:.4f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'formation_energy_optimized.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {plot_path}")

# =========================
# SAVE TABLES
# =========================
results_path = OUTPUT_DIR / 'model_comparison.csv'
fi_path = OUTPUT_DIR / 'feature_importance.csv'
pred_path = OUTPUT_DIR / 'predictions.csv'

results_df.to_csv(results_path, index=False)
fi_df.to_csv(fi_path, index=False)

pred_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': best_pred,
    'Error': (y_test.values - best_pred),
    'Abs_Error': np.abs(y_test.values - best_pred)
})
pred_df.to_csv(pred_path, index=False)

print("✓ Saved:", results_path)
print("✓ Saved:", fi_path)
print("✓ Saved:", pred_path)

# =========================
# FINAL SUMMARY
# =========================
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}\n")
print(f"Best Model: {best_name}")
print(f"Test R²: {best_r2:.4f}")
print(f"Test MAE: {best_mae:.4f} eV/atom")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, best_pred)):.4f} eV/atom")
print(f"\nKey Success Factors:")
print(f"   ✓ Physically meaningful interaction features")
print(f"   ✓ Perovskite-specific descriptors (tolerance factors)")
print(f"   ✓ Custom features from symbolic regression insights")
print(f"   ✓ Electronic-geometric coupling terms")