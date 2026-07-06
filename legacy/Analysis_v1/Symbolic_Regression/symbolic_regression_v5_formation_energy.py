import pandas as pd
import numpy as np
from mendeleev import element
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.utils.validation import validate_data
from tqdm import tqdm

# --- Compatibility shim for scikit-learn 1.7+ ---
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

# ==========================================
# 1. LOAD DATA
# ==========================================
DATA_PATH = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Data\true and non true double perovskite sort.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("Fetching Electronegativity values from Mendeleev database...")

def get_en(el_symbol):
    if pd.isna(el_symbol) or str(el_symbol).strip() == '':
        return np.nan
    try:
        return element(str(el_symbol).strip()).electronegativity()
    except:
        return np.nan

df['EN_A'] = df['element1'].apply(get_en)
df['EN_B'] = df['element3'].apply(get_en)
df['EN_Bprime'] = df['element4'].apply(get_en)
df['EN_O'] = 3.44  # fixed oxygen electronegativity

O_RADIUS = 1.40
df['bond_length_A_O'] = pd.to_numeric(df['metallic radius1'], errors='coerce') + O_RADIUS
df['bond_length_B_O'] = pd.to_numeric(df['metallic radius 3'], errors='coerce') + O_RADIUS
df['bond_length_Bprime_O'] = pd.to_numeric(df['metallic radius4'], errors='coerce') + O_RADIUS

input_features = [
    'EN_A', 'EN_B', 'EN_Bprime', 'EN_O',
    'bond_length_A_O', 'bond_length_B_O', 'bond_length_Bprime_O'
]

target = 'Formation Energy (eV/atom)'

# Convert target to numeric
df[target] = pd.to_numeric(df[target], errors='coerce')

# Clean
df_clean = df.dropna(subset=[target]).copy()
df_clean[input_features] = df_clean[input_features].fillna(0)

X = df_clean[input_features].copy()
y = df_clean[target].copy()

print(f"\n{'='*50}\n3-STAGE MODEL: LINEAR BASELINE + SYMBOLIC RESIDUAL\n{'='*50}")

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. STAGE 1: LINEAR BASELINE
#    Use Ridge because it is stable with correlated features
# ==========================================
baseline_model = Ridge(alpha=1.0, random_state=42)
baseline_model.fit(X_train, y_train)

base_train_pred = baseline_model.predict(X_train)
base_test_pred = baseline_model.predict(X_test)

baseline_r2 = r2_score(y_test, base_test_pred)
print(f"\n📊 Baseline Ridge R²: {baseline_r2:.4f}")

# ==========================================
# 5. STAGE 2: SYMBOLIC REGRESSION ON RESIDUALS
# ==========================================
resid_train = y_train - base_train_pred
resid_test = y_test - base_test_pred

# Standardize residuals using TRAIN stats only
resid_mean = resid_train.mean()
resid_std = resid_train.std()

if resid_std == 0 or np.isnan(resid_std):
    resid_std = 1.0

resid_train_scaled = (resid_train - resid_mean) / resid_std

TOTAL_GENERATIONS = 100

est_gp = SymbolicRegressor(
    population_size=3000,
    generations=1,
    warm_start=True,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=0,
    parsimony_coefficient=0.001,
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'abs')
)

for gen in tqdm(range(TOTAL_GENERATIONS), desc="Evolving residual", unit="gen"):
    est_gp.set_params(generations=gen + 1)
    est_gp.fit(X_train, resid_train_scaled)

# Predict residuals, then invert scaling
resid_pred_scaled = est_gp.predict(X_test)
resid_pred = resid_pred_scaled * resid_std + resid_mean

# ==========================================
# 6. FINAL PREDICTION
# ==========================================
final_pred = base_test_pred + resid_pred
final_r2 = r2_score(y_test, final_pred)

print(f"\n📊 Final R² Accuracy (baseline + symbolic residual): {final_r2:.4f}")

# ==========================================
# 7. PRINT BASELINE EQUATION
# ==========================================
baseline_intercept = baseline_model.intercept_
baseline_coefs = baseline_model.coef_

baseline_terms = [f"({baseline_intercept:.6f})"]
for coef, feat in zip(baseline_coefs, input_features):
    if abs(coef) > 1e-10:
        baseline_terms.append(f"({coef:.6f})*[{feat}]")

baseline_equation = " + ".join(baseline_terms)

print("\n📐 Baseline Ridge Equation:")
print(f"{target} ≈ {baseline_equation}")

# ==========================================
# 8. PRINT SYMBOLIC RESIDUAL EQUATION
# ==========================================
equation = str(est_gp._program)
for i, feature in reversed(list(enumerate(input_features))):
    equation = equation.replace(f"X{i}", f"[{feature}]")

print("\n📐 Residual Equation:")
print(f"residual ≈ {equation}")

print("\n📐 Final Model:")
print(f"{target} ≈ baseline_linear_model + residual")