import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import validate_data

# Compatibility shim for scikit-learn 1.7+
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

# 1. Load Data
DATA_PATH = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Data\true and non true double perovskite sort.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')

# 2. STRICT SEPARATION OF VARIABLES
# These are the design choices you make in the lab
input_features = [
    'metallic radius1', 'metallic radius 2', 'metallic radius 3', 'metallic radius4'
]

# These are the physical outcomes you want to predict
target_properties = [
     'Total Magnetization (Î¼B)','Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)'
]

# Clean data
all_cols = input_features + target_properties
for col in all_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df_clean = df.dropna(subset=target_properties).copy()
df_clean[input_features] = df_clean[input_features].fillna(0)

print(f"{'='*50}\nSYMBOLIC REGRESSION: FORMULA DISCOVERY\n{'='*50}")

# 3. Predict Targets using ONLY Input Features
X = df_clean[input_features]

for target in target_properties:
    print(f"\n🎯 Target Property: {target}")
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the "Automated Physicist"
    est_gp = SymbolicRegressor(
        population_size=2000,       # Increased population for better genetic diversity
        generations=40,             # Let it evolve for longer
        stopping_criteria=0.01,
        p_crossover=0.7, 
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, 
        p_point_mutation=0.1,
        max_samples=0.9, 
        verbose=0,
        parsimony_coefficient=0.0005, # DRASTICALLY lowered so it stops predicting flat constants
        random_state=42,
        function_set=(
            'add', 'sub', 'mul', 'div',
            'sqrt', 'abs', 'sin', 'cos',
            'tan', 'log', 'inv', 'max',
            'min', 'neg'
        )
    )
    
    # Fit and pass the actual feature names for readable output
    est_gp.fit(X_train, y_train)
    y_pred = est_gp.predict(X_test)
    
    # Print the results
    print(f"📊 R² Accuracy: {r2_score(y_test, y_pred):.3f}")
    
    # We must format the program string to use the actual column names
    equation = str(est_gp._program)
    for i, feature in reversed(list(enumerate(input_features))):
        equation = equation.replace(f"X{i}", f"[{feature}]")
        
    print(f"📐 Discovered Equation:\n{target} ≈ {equation}")