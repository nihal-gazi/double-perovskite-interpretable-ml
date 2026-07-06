import pandas as pd
import numpy as np
from pathlib import Path
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.validation import validate_data
from tqdm import tqdm

# --- scikit-learn compatibility ---
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

# =========================
# 1. SETUP & EN DICT (Lightning Fast)
# =========================
INPUT_XLSX = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Data\true and non true double perovskite sort.xlsx")

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

def get_en(el):
    if pd.isna(el) or str(el).strip() == '': return np.nan
    return EN_DICT.get(str(el).strip(), np.nan)

# =========================
# 2. FEATURE ENGINEERING (From your GB Model)
# =========================
df = pd.read_excel(INPUT_XLSX, sheet_name='approx true double perovskite')

df['EN_A'] = df['element1'].apply(get_en)
df['EN_B'] = df['element3'].apply(get_en)
df['EN_Bprime'] = df['element4'].apply(get_en)

O_RADIUS = 1.40
eps = 1e-9
df['d_AO'] = pd.to_numeric(df['metallic radius1'], errors='coerce') + O_RADIUS
df['d_BO'] = pd.to_numeric(df['metallic radius 3'], errors='coerce') + O_RADIUS
df['d_BprimeO'] = pd.to_numeric(df['metallic radius4'], errors='coerce') + O_RADIUS

# The critical macro-features your GB model loved
df['EN_avg'] = (df['EN_B'] + df['EN_Bprime']) / 2
df['d_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
df['d_B_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
df['octahedral_factor'] = df['d_B_avg'] / (df['d_AO'] + eps)
df['tolerance_factor'] = df['d_AO'] / (np.sqrt(2) * df['d_B_avg'] + eps)
df['EN_ratio_B_A'] = df['EN_B'] / (df['EN_A'] + eps)

target = 'Formation Energy (eV/atom)'
df[target] = pd.to_numeric(df[target], errors='coerce')

# We only feed the TOP features into the Symbolic Regressor to force a clean equation
gb_top_features = [
    'EN_avg', 'd_avg', 'octahedral_factor', 'tolerance_factor', 'EN_ratio_B_A'
]

df_clean = df.dropna(subset=[target]).copy()
for col in gb_top_features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

X = df_clean[gb_top_features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 3. GB-INFORMED SYMBOLIC REGRESSION
# =========================
print(f"\n{'='*60}\nDISTILLING GB LOGIC INTO A MATHEMATICAL FORMULA\n{'='*60}")

# Lowering parsimony to let it build a robust formula, adding sqrt back
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
    parsimony_coefficient=0.0005, # Loosened so it can match GB complexity
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'sqrt') 
)

TOTAL_GENERATIONS = 150 # It will run incredibly fast without Mendeleev

for gen in tqdm(range(TOTAL_GENERATIONS), desc="Evolving Formula", unit="gen"):
    est_gp.set_params(generations=gen + 1)
    est_gp.fit(X_train, y_train)

y_pred = est_gp.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Final Formula Accuracy (R²): {r2:.3f}")

# Format equation cleanly
equation = str(est_gp._program)
for i, feature in reversed(list(enumerate(gb_top_features))):
    equation = equation.replace(f"X{i}", f"[{feature}]")
    
print(f"\n📐 FINAL DISCOVERED EQUATION:\nFormation Energy ≈ {equation}")