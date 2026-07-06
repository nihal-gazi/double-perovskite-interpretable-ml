import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_and_evaluate_baseline_models(df_merged):
    # 1. Linear features (19 features used in combined_linear_regression.py)
    linear_features = [
        'Val_A', 'Val_B', 'Val_Bprime', 'EN_A', 'EN_B', 'EN_Bprime',
        'Shannon_A', 'Shannon_B', 'Shannon_Bprime', 
        'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 'EN_diff', 'Val_avg',
        '3D_Volume_Per_Atom', '3D_Density',
        'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
    ]
    
    # 2. GB features for Formation Energy (27 features from legacy ridge_gb_v1.py)
    # We construct them here to ensure exact reproducibility
    df_gb = df_merged.copy()
    eps = 1e-9
    df_gb['EN_O'] = 3.44
    df_gb['EN_B_minus_A'] = df_gb['EN_B'] - df_gb['EN_A']
    df_gb['EN_Bprime_minus_A'] = df_gb['EN_Bprime'] - df_gb['EN_A']
    df_gb['EN_Bprime_minus_B'] = df_gb['EN_Bprime'] - df_gb['EN_B']
    df_gb['EN_ratio_Bprime_A'] = df_gb['EN_Bprime'] / (df_gb['EN_A'] + eps)
    df_gb['EN_ratio_B_A'] = df_gb['EN_B'] / (df_gb['EN_A'] + eps)
    
    df_gb['d_BprimeO_minus_BO'] = df_gb['d_BprimeO'] - df_gb['d_BO']
    df_gb['d_AO_minus_BO'] = df_gb['d_AO'] - df_gb['d_BO']
    df_gb['d_ratio_BprimeO_BO'] = df_gb['d_BprimeO'] / (df_gb['d_BO'] + eps)
    df_gb['d_ratio_AO_BO'] = df_gb['d_AO'] / (df_gb['d_BO'] + eps)
    
    df_gb['d_B_avg'] = (df_gb['d_BO'] + df_gb['d_BprimeO']) / 2
    df_gb['tolerance_factor'] = df_gb['d_AO'] / (np.sqrt(2) * df_gb['d_B_avg'] + eps)
    df_gb['octahedral_factor'] = df_gb['d_B_avg'] / (df_gb['d_AO'] + eps)
    df_gb['distortion_idx'] = np.abs(df_gb['d_BO'] - df_gb['d_BprimeO']) / (df_gb['d_B_avg'] + eps)
    
    df_gb['custom_tolerance'] = (df_gb['EN_Bprime'] / (df_gb['EN_A'] + eps)) / (df_gb['d_BO'] / (df_gb['d_BprimeO'] + eps) + eps)
    df_gb['EN_d_mismatch'] = df_gb['EN_Bprime'] * df_gb['d_BprimeO'] - df_gb['EN_B'] * df_gb['d_BO']
    df_gb['bond_strength_B'] = df_gb['EN_B'] / (df_gb['d_BO']**2 + eps)
    df_gb['bond_strength_Bprime'] = df_gb['EN_Bprime'] / (df_gb['d_BprimeO']**2 + eps)
    df_gb['EN_std'] = np.abs(df_gb['EN_B'] - df_gb['EN_Bprime'])
    df_gb['d_std'] = np.abs(df_gb['d_BO'] - df_gb['d_BprimeO'])
    
    fe_gb_features = [
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
    
    # 3. GB features for other targets (6 features from oracle_chgnet.py)
    other_gb_features = [
        'Tolerance_Factor', 'Octahedral_Mismatch', 'EN_avg', 
        'CHGNet_Energy', 'CHGNet_Net_Magmom', 'CHGNet_Abs_Magmom'
    ]
    
    targets = [
        'Formation Energy (eV/atom)',
        'Total Magnetization (uB)',
        'Band Gap (eV)',
        'Energy Above Hull (eV)'
    ]
    
    results = {}
    
    for target in targets:
        # Linear models split
        df_clean_lin = df_merged.dropna(subset=linear_features + [target]).copy()
        X_lin = df_clean_lin[linear_features]
        y_lin = df_clean_lin[target]
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_l_scaled = scaler.fit_transform(X_train_l)
        X_test_l_scaled = scaler.transform(X_test_l)
        
        # Fit OLS
        ols = LinearRegression()
        ols.fit(X_train_l_scaled, y_train_l)
        y_pred_ols = ols.predict(X_test_l_scaled)
        ols_r2 = r2_score(y_test_l, y_pred_ols)
        ols_mae = mean_absolute_error(y_test_l, y_pred_ols)
        
        # Fit Ridge
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 100))
        ridge.fit(X_train_l_scaled, y_train_l)
        y_pred_ridge = ridge.predict(X_test_l_scaled)
        ridge_r2 = r2_score(y_test_l, y_pred_ridge)
        ridge_mae = mean_absolute_error(y_test_l, y_pred_ridge)
        
        # Fit Lasso
        lasso = LassoCV(alphas=np.logspace(-3, 3, 100), max_iter=10000)
        lasso.fit(X_train_l_scaled, y_train_l)
        y_pred_lasso = lasso.predict(X_test_l_scaled)
        lasso_r2 = r2_score(y_test_l, y_pred_lasso)
        lasso_mae = mean_absolute_error(y_test_l, y_pred_lasso)
        
        # Fit Gradient Boosting
        if target == 'Formation Energy (eV/atom)':
            df_clean_gb = df_gb.dropna(subset=fe_gb_features + [target]).copy()
            X_gb = df_clean_gb[fe_gb_features]
            y_gb = df_clean_gb[target]
            X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        else:
            df_clean_gb = df_merged.dropna(subset=other_gb_features + [target]).copy()
            X_gb = df_clean_gb[other_gb_features]
            y_gb = df_clean_gb[target]
            X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
            
        gb.fit(X_train_g, y_train_g)
        y_pred_gb = gb.predict(X_test_g)
        gb_r2 = r2_score(y_test_g, y_pred_gb)
        gb_mae = mean_absolute_error(y_test_g, y_pred_gb)
        
        results[target] = {
            'OLS': {'R2': ols_r2, 'MAE': ols_mae},
            'Ridge': {'R2': ridge_r2, 'MAE': ridge_mae},
            'Lasso': {'R2': lasso_r2, 'MAE': lasso_mae},
            'GB': {'R2': gb_r2, 'MAE': gb_mae},
            'Lasso_Model': lasso,
            'Lasso_Scaler': scaler,
            'Lasso_Features': linear_features,
            'GB_Model': gb,
            'GB_Split': (X_train_g, X_test_g, y_train_g, y_test_g, y_pred_gb)
        }
        
    return results
