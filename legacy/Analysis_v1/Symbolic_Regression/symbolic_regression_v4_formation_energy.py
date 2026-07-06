import pandas as pd
import numpy as np
from mendeleev import element
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
        return 0
    try:
        return element(str(el_symbol).strip()).electronegativity()
    except:
        return 0

df['EN_A'] = df['element1'].apply(get_en)
df['EN_B'] = df['element3'].apply(get_en)
df['EN_Bprime'] = df['element4'].apply(get_en)
df['EN_O'] = 3.44

O_RADIUS = 1.40
df['bond_length_A_O'] = df['metallic radius1'] + O_RADIUS
df['bond_length_B_O'] = df['metallic radius 3'] + O_RADIUS
df['bond_length_Bprime_O'] = df['metallic radius4'] + O_RADIUS

input_features = [
    'EN_A', 'EN_B', 'EN_Bprime', 'EN_O',
    'bond_length_A_O', 'bond_length_B_O', 'bond_length_Bprime_O'
]

target_properties = ['Formation Energy (eV/atom)']

all_cols = input_features + target_properties
for col in all_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=target_properties).copy()
df_clean[input_features] = df_clean[input_features].fillna(0)

print(f"\n{'='*50}\nRESIDUAL SYMBOLIC REGRESSION\n{'='*50}")

# ==========================================
# 3. BASELINE GUESS
#    (your hand-crafted starting point)
# ==========================================
def baseline_guess(dataframe):
    eps = 1e-9

    EN_A = dataframe['EN_A'].to_numpy()
    EN_B = dataframe['EN_B'].to_numpy()
    EN_Bprime = dataframe['EN_Bprime'].to_numpy()
    EN_O = dataframe['EN_O'].to_numpy()

    d_BO = dataframe['bond_length_B_O'].to_numpy()
    d_Bprime_O = dataframe['bond_length_Bprime_O'].to_numpy()

    EN_A_safe = np.where(np.abs(EN_A) < eps, eps, EN_A)
    d_BO_safe = np.where(np.abs(d_BO) < eps, eps, d_BO)

    guess = (
        EN_B
        - d_Bprime_O
        + 0.892
        + 0.382 * (EN_Bprime / EN_A_safe) * (d_Bprime_O / d_BO_safe)
        - 0.955 * EN_O
    )
    return pd.Series(guess, index=dataframe.index)

# ==========================================
# 4. SYMBOLIC REGRESSION ON RESIDUALS
# ==========================================
TOTAL_GENERATIONS = 100

for target in target_properties:
    print(f"\n🎯 Target Property: {target}")

    # Baseline guess
    df_clean['baseline_guess'] = baseline_guess(df_clean)

    # Residual target = actual - baseline
    df_clean['residual_target'] = df_clean[target] - df_clean['baseline_guess']

    X = df_clean[input_features]
    y_resid = df_clean['residual_target']
    y_true = df_clean[target]
    y_base = df_clean['baseline_guess']

    X_train, X_test, y_train_resid, y_test_resid, y_train_true, y_test_true, base_train, base_test = train_test_split(
        X, y_resid, y_true, y_base, test_size=0.2, random_state=42
    )

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

    # Evolve the residual model
    for gen in tqdm(range(TOTAL_GENERATIONS), desc=f"Evolving residual for {target}", unit="gen"):
        est_gp.set_params(generations=gen + 1)
        est_gp.fit(X_train, y_train_resid)

    # Predict residuals, then add baseline back
    resid_pred = est_gp.predict(X_test)
    final_pred = base_test.to_numpy() + resid_pred

    r2 = r2_score(y_test_true, final_pred)
    print(f"\n📊 Final R² Accuracy (baseline + evolved residual): {r2:.3f}")

    # Show residual equation
    equation = str(est_gp._program)
    for i, feature in reversed(list(enumerate(input_features))):
        equation = equation.replace(f"X{i}", f"[{feature}]")

    print(f"\n📐 Residual Equation:\nresidual ≈ {equation}")

    print("\n📐 Final Model:")
    print("Formation Energy ≈ baseline_guess + residual")