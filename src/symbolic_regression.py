import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils.validation import validate_data
from gplearn.genetic import SymbolicRegressor

# Scikit-learn 1.6+ compatibility patch for gplearn
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        return validate_data(self, X, y, reset=reset, validate_separately=validate_separately, **check_params)
    SymbolicRegressor._validate_data = _validate_data

def fit_symbolic_regressions(df_merged):
    # 1. Setup features for Formation Energy (11 features from legacy code)
    fe_features = [
        'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
        'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
        'Tolerance_Factor', 'Octahedral_Mismatch'
    ]
    
    # 2. Setup features for other targets (6 features from legacy chgnet code)
    chgnet_features = [
        'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 
        'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
    ]
    
    targets = {
        'Formation Energy (eV/atom)': {
            'features': fe_features,
            'pop_size': 4000,
            'gens': 60,
            'parsimony': 0.001
        },
        'Total Magnetization (uB)': {
            'features': chgnet_features,
            'pop_size': 3000,
            'gens': 50,
            'parsimony': 0.002
        },
        'Band Gap (eV)': {
            'features': chgnet_features,
            'pop_size': 3000,
            'gens': 50,
            'parsimony': 0.002
        },
        'Energy Above Hull (eV)': {
            'features': chgnet_features,
            'pop_size': 3000,
            'gens': 50,
            'parsimony': 0.002
        }
    }
    
    results = {}
    
    for target, params in targets.items():
        print(f"Running Symbolic Regression for {target}...")
        features = params['features']
        
        # Clean NaNs
        clean_cols = features + [target]
        d_clean = df_merged.dropna(subset=clean_cols).copy()
        d_clean[features] = d_clean[features].fillna(0)
        
        X = d_clean[features]
        y = d_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        est_gp = SymbolicRegressor(
            population_size=params['pop_size'],
            generations=params['gens'],
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            parsimony_coefficient=params['parsimony'],
            random_state=42,
            function_set=('add', 'sub', 'mul', 'div', 'abs')
        )
        
        est_gp.fit(X_train, y_train)
        
        y_pred = est_gp.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Format discovered equation cleanly
        equation = str(est_gp._program)
        for i, feature in reversed(list(enumerate(features))):
            equation = equation.replace(f"X{i}", f"[{feature}]")
            
        results[target] = {
            'R2': r2,
            'MAE': mae,
            'Equation': equation,
            'Model': est_gp
        }
        print(f"   Test R^2: {r2:.4f} | Test MAE: {mae:.4f}")
        print(f"   Discovered Equation: {equation}\n")
        
    return results
