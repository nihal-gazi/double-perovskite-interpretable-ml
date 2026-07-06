import os
import pandas as pd
import numpy as np
from pathlib import Path
from mendeleev import element
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.validation import validate_data
from tqdm import tqdm

# --- scikit-learn 1.7+ compatibility ---
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

# ==========================================
# 1. SETUP & PATHS
# ==========================================
BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
CACHE_PATH = BASE_DIR / "Data" / "mendeleev_cached_features.csv"

# ==========================================
# 2. CACHED DATA LOADING OR FETCHING
# ==========================================
if CACHE_PATH.exists():
    print(f"⚡ Loading chemistry data instantly from cache: {CACHE_PATH}")
    df_clean = pd.read_csv(CACHE_PATH)
    
else:
    print("⏳ Cache not found. Querying Mendeleev database (this will take a few minutes)...")
    df = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')
    
    # Safe fetchers for Mendeleev
    def get_chem_props(el_symbol):
        if pd.isna(el_symbol) or str(el_symbol).strip() == '':
            return np.nan, np.nan, np.nan
        try:
            el = element(str(el_symbol).strip())
            
            # Get Valence and EN
            val = el.nvalence()
            en = el.electronegativity()
            
            # Safely average Shannon Ionic Radii (if available), else fallback to covalent
            if el.ionic_radii:
                # Convert pm to Å (divide by 100 if mendeleev returns pm)
                ionic_r = np.mean([ir.ionic_radius for ir in el.ionic_radii if ir.ionic_radius is not None])
                if ionic_r > 10: ionic_r /= 100 # Rough conversion if in pm
            else:
                ionic_r = el.covalent_radius_pyykko / 100 if el.covalent_radius_pyykko else np.nan
                
            return val, en, ionic_r
        except:
            return np.nan, np.nan, np.nan

    # Apply to A, B, and B' sites
    print("Fetching A-site properties...")
    df['Val_A'], df['EN_A'], df['Shannon_A'] = zip(*df['element1'].apply(get_chem_props))
    print("Fetching B-site properties...")
    df['Val_B'], df['EN_B'], df['Shannon_B'] = zip(*df['element3'].apply(get_chem_props))
    print("Fetching B'-site properties...")
    df['Val_Bprime'], df['EN_Bprime'], df['Shannon_Bprime'] = zip(*df['element4'].apply(get_chem_props))
    
    # Oxygen Constants
    O_SHANNON = 1.40 
    
    # 3. ADVANCED DESCRIPTORS (Tolerance, Mismatch)
    print("Calculating Crystallographic Features...")
    eps = 1e-9
    df['d_AO'] = df['Shannon_A'] + O_SHANNON
    df['d_BO'] = df['Shannon_B'] + O_SHANNON
    df['d_BprimeO'] = df['Shannon_Bprime'] + O_SHANNON
    
    df['d_B_avg'] = (df['d_BO'] + df['d_BprimeO']) / 2
    
    # The true Goldschmidt Tolerance Factor
    df['Tolerance_Factor'] = df['d_AO'] / (np.sqrt(2) * df['d_B_avg'] + eps)
    
    # Octahedral Mismatch (How different are the B and B' shapes?)
    df['Octahedral_Mismatch'] = np.abs(df['d_BO'] - df['d_BprimeO']) / (df['d_B_avg'] + eps)

    # Clean and Cache
    target = 'Formation Energy (eV/atom)'
    # We must explicitly keep the categorical Space Group for future use, but we won't use it in the math formula
    keep_cols = [
        'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
        'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
        'Tolerance_Factor', 'Octahedral_Mismatch', target, 'Space Group'
    ]
    df_clean = df.dropna(subset=[target]).copy()
    for col in keep_cols:
        if col != 'Space Group':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
    df_clean = df_clean[keep_cols]
    df_clean.to_csv(CACHE_PATH, index=False)
    print(f"✅ Data cached successfully to: {CACHE_PATH}")

# ==========================================
# 4. SYMBOLIC REGRESSION (Formula Generation)
# ==========================================
print(f"\n{'='*50}\nGENERATING MATHEMATICAL FORMULA\n{'='*50}")

# Note: Space Group is categorical, math equations can't use "Fm-3m". We exclude it from the algebraic search.
input_features = [
    'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
    'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
    'Tolerance_Factor', 'Octahedral_Mismatch'
]
target = 'Formation Energy (eV/atom)'

X = df_clean[input_features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the automated physicist
est_gp = SymbolicRegressor(
    population_size=4000,         # Massive population to handle new features
    generations=1,                
    warm_start=True,              
    stopping_criteria=0.01,
    p_crossover=0.7, 
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05, 
    p_point_mutation=0.1,
    max_samples=0.9, 
    verbose=0,
    parsimony_coefficient=0.001,  # Keeps the formula from becoming 5 pages long
    random_state=42,
    function_set=('add', 'sub', 'mul', 'div', 'abs') # Removed sqrt to keep the equation looking like standard chemistry
)

TOTAL_GENERATIONS = 60 # Evolve for 60 generations

for gen in tqdm(range(TOTAL_GENERATIONS), desc="Evolving Formula", unit="gen"):
    est_gp.set_params(generations=gen + 1)
    est_gp.fit(X_train, y_train)

y_pred = est_gp.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Formula Accuracy (R²): {r2:.3f}")

# Format equation cleanly
equation = str(est_gp._program)
for i, feature in reversed(list(enumerate(input_features))):
    equation = equation.replace(f"X{i}", f"[{feature}]")
    
print(f"\n📐 FINAL DISCOVERED EQUATION:\nFormation Energy ≈ {equation}")