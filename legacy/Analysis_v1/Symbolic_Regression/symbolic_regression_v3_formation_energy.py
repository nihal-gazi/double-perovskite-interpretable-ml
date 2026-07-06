import pandas as pd
import numpy as np
from mendeleev import element
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.validation import validate_data
from tqdm import tqdm # <-- The magic progress bar

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
# 2. FEATURE ENGINEERING (EN & Bond Lengths)
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

target_properties = [
    'Formation Energy (eV/atom)'

]

all_cols = input_features + target_properties
for col in all_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df_clean = df.dropna(subset=target_properties).copy()
df_clean[input_features] = df_clean[input_features].fillna(0)

print(f"\n{'='*50}\nSYMBOLIC REGRESSION: ELECTRONIC & STRUCTURAL DISCOVERY\n{'='*50}")

# ==========================================
# 3. SYMBOLIC REGRESSION PIPELINE (w/ TQDM)
# ==========================================
X = df_clean[input_features]
TOTAL_GENERATIONS = 100 # Set your target generations here

for target in target_properties:
    print(f"\n🎯 Target Property: {target}")
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Regressor with warm_start=True
    est_gp = SymbolicRegressor(
        population_size=3000,         
        generations=1,                # Start at 1, we will manually step it forward
        warm_start=True,              # CRITICAL: Keeps the population alive between fit() calls
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
    
    # The TQDM Manual Evolution Loop!
    for gen in tqdm(range(TOTAL_GENERATIONS), desc=f"Evolving {target}", unit="gen"):
        est_gp.set_params(generations=gen + 1) # Tell it to compute the next generation
        est_gp.fit(X_train, y_train)
    
    y_pred = est_gp.predict(X_test)
    
    # Accuracy Score
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 R² Accuracy: {r2:.3f}")
    
    # Format the equation to be human-readable
    equation = str(est_gp._program)
    for i, feature in reversed(list(enumerate(input_features))):
        equation = equation.replace(f"X{i}", f"[{feature}]")
        
    print(f"📐 Discovered Equation:\n{target} ≈ {equation}")