import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to python path if needed
sys.path.append(str(Path(__file__).parent))

from src.config import MODELS_DIR, RESULTS_DIR
from src.data_processing import load_and_preprocess_data
from src.VAE import train_and_project_vae
from src.baseline_models import train_and_evaluate_baseline_models
from src.symbolic_regression import fit_symbolic_regressions
from src.plots import (
    generate_correlation_heatmap,
    generate_pairplot,
    generate_tsne_plots,
    generate_gbr_diagnostics
)

def parse_args():
    parser = argparse.ArgumentParser(description="Double Perovskite Machine Learning Pipeline")
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Bypass heavy VAE training by loading pre-calculated t-SNE embeddings."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    # Replaced unicode emoji with plain text log identifier to prevent encoding crashes
    print("[RUNNING] DOUBLE PEROVSKITE PIPELINE")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # STAGE 1: LOAD & PREPROCESS DATA
    # -------------------------------------------------------------------------
    print("[STAGE 1] Loading and merging datasets...")
    try:
        df_merged = load_and_preprocess_data()
        print(f"[OK] Data loaded successfully. Total samples: {len(df_merged)}")
    except Exception as e:
        print(f"Error during Stage 1 data loading: {e}", file=sys.stderr)
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # STAGE 2: VAE & t-SNE DIMENSION REDUCTION
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Running VAE and t-SNE visualization...")
    if args.fast:
        print("[INFO] Fast mode active: Loading pre-calculated t-SNE embeddings...")
        tsne_path = MODELS_DIR / "tSNE" / "tsne_embeddings_2d.pkl"
        if not tsne_path.exists():
            print(f"Error: Pre-calculated t-SNE embeddings not found at {tsne_path}. Please run without --fast first.")
            sys.exit(1)
        df_tsne = pd.read_pickle(tsne_path)
    else:
        df_tsne = train_and_project_vae(epochs=1000)
        
    # Generate t-SNE scatter plots
    generate_tsne_plots(df_tsne)
    
    # -------------------------------------------------------------------------
    # EXPLORATORY PLOTS
    # -------------------------------------------------------------------------
    print("\n[PLOTS] Generating exploratory correlation heatmap and pairplots...")
    # Setup engineered features for exploration
    df_merged['EN_B_minus_A'] = df_merged['EN_B'] - df_merged['EN_A']
    df_merged['EN_ratio_B_A'] = df_merged['EN_B'] / (df_merged['EN_A'] + 1e-9)
    df_merged['octahedral_factor'] = df_merged['d_avg'] / (df_merged['d_AO'] + 1e-9)
    df_merged['distortion_idx'] = np.abs(df_merged['d_BO'] - df_merged['d_BprimeO']) / (df_merged['d_avg'] + 1e-9)
    df_merged['custom_tolerance'] = (df_merged['EN_Bprime'] / (df_merged['EN_A'] + 1e-9)) / (df_merged['d_BO'] / (df_merged['d_BprimeO'] + 1e-9) + 1e-9)
    df_merged['bond_strength_Bprime'] = df_merged['EN_Bprime'] / (df_merged['d_BprimeO']**2 + 1e-9)
    
    generate_correlation_heatmap(df_merged)
    generate_pairplot(df_merged)
    
    # -------------------------------------------------------------------------
    # STAGE 3: BASELINE REGRESSION MODELS
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Training baseline OLS, Ridge, Lasso, and GB Trees...")
    baseline_results = train_and_evaluate_baseline_models(df_merged)
    
    # Save diagnostic plot for Formation Energy (as in Figure 7)
    fe_res = baseline_results['Formation Energy (eV/atom)']
    X_train_g, X_test_g, y_train_g, y_test_g, y_pred_gb = fe_res['GB_Split']
    # 27 features from ridge_gb_v1.py
    fe_features = [
        'EN_A', 'EN_B', 'EN_Bprime', 'd_AO', 'd_BO', 'd_BprimeO',
        'EN_B_minus_A', 'EN_Bprime_minus_A', 'EN_Bprime_minus_B',
        'EN_ratio_Bprime_A', 'EN_ratio_B_A', 'd_BprimeO_minus_BO', 'd_AO_minus_BO',
        'd_ratio_BprimeO_BO', 'd_ratio_AO_BO', 'd_B_avg', 'tolerance_factor', 'octahedral_factor',
        'distortion_idx', 'custom_tolerance', 'EN_d_mismatch', 'bond_strength_B', 'bond_strength_Bprime',
        'EN_avg', 'EN_std', 'd_avg', 'd_std'
    ]
    generate_gbr_diagnostics(
        y_test_g, y_pred_gb, 
        fe_features, 
        fe_res['GB_Model'].feature_importances_, 
        'Formation Energy (eV/atom)'
    )
    
    # Print Table 2: Model Performance
    print("\n" + "="*90)
    print("TABLE 2: MODEL PERFORMANCE (R^2) AND MAE")
    print("="*90)
    print(f"{'Target Property':<30} | {'OLS R2 (MAE)':<18} | {'Ridge R2 (MAE)':<18} | {'Lasso R2 (MAE)':<18} | {'GB R2 (MAE)':<18}")
    print("-"*90)
    for target, res in baseline_results.items():
        print(f"{target:<30} | "
              f"{res['OLS']['R2']:.4f} ({res['OLS']['MAE']:.4f}) | "
              f"{res['Ridge']['R2']:.4f} ({res['Ridge']['MAE']:.4f}) | "
              f"{res['Lasso']['R2']:.4f} ({res['Lasso']['MAE']:.4f}) | "
              f"{res['GB']['R2']:.4f} ({res['GB']['MAE']:.4f})")
    print("="*90 + "\n")
    
    # -------------------------------------------------------------------------
    # STAGE 4: SYMBOLIC REGRESSION
    # -------------------------------------------------------------------------
    print("[STAGE 4] Discovering physical equations via Genetic Programming...")
    symbolic_results = fit_symbolic_regressions(df_merged)
    
    # Print Table 3: Modeling Approach Comparison
    print("\n" + "="*70)
    print("TABLE 3: PERFORMANCE METRICS (R^2) COMPARISON")
    print("="*70)
    print(f"{'Target Property':<30} | {'Best Linear R2':<18} | {'GBR R2 Ceiling':<15} | {'Symbolic R2':<12}")
    print("-"*70)
    for target in baseline_results.keys():
        best_lin_model = 'Lasso' if baseline_results[target]['Lasso']['R2'] > baseline_results[target]['OLS']['R2'] else 'OLS'
        best_lin_r2 = baseline_results[target][best_lin_model]['R2']
        gbr_r2 = baseline_results[target]['GB']['R2']
        sym_r2 = symbolic_results[target]['R2']
        print(f"{target:<30} | {best_lin_r2:.4f} ({best_lin_model:<5}) | {gbr_r2:.4f}          | {sym_r2:.4f}")
    print("="*70 + "\n")
    
    # Print Table 4: Discovered Algebraic Equations
    print("\n" + "="*110)
    print("TABLE 4: BEST DISCOVERED PHYSICAL EQUATIONS AND MODEL PARAMETERS")
    print("="*110)
    print(f"{'Target Property':<30} | {'Best R2':<8} | {'Method':<20} | {'Discovered Equation':<45}")
    print("-"*110)
    
    # 1. Formation Energy (Lasso Equation)
    fe_lasso_res = baseline_results['Formation Energy (eV/atom)']
    # Reconstruct the Lasso equation string as formatted in combined_linear_regression.py
    coefs = fe_lasso_res['Lasso_Model'].coef_
    intercept = fe_lasso_res['Lasso_Model'].intercept_
    scaler = fe_lasso_res['Lasso_Scaler']
    unscaled_coefs = coefs / np.sqrt(scaler.var_)
    unscaled_intercept = intercept - np.sum(coefs * scaler.mean_ / np.sqrt(scaler.var_))
    sig_df = pd.DataFrame({
        'Feature': fe_lasso_res['Lasso_Features'],
        'Coef': unscaled_coefs,
        'Abs': np.abs(coefs)
    }).sort_values('Abs', ascending=False)
    
    eq_terms = []
    for _, row in sig_df.head(4).iterrows():
        sign = "+" if row['Coef'] >= 0 else "-"
        eq_terms.append(f"{sign} {abs(row['Coef']):.4f} * {row['Feature']}")
    sign_int = "+" if unscaled_intercept >= 0 else "-"
    fe_eq = " ".join(eq_terms) + f" {sign_int} {abs(unscaled_intercept):.4f}"
    print(f"{'Formation Energy (Ef)':<30} | {fe_lasso_res['Lasso']['R2']:.4f} | {'Lasso Regression':<20} | {fe_eq:<45}")
    
    # 2. Total Magnetization, Energy Above Hull, Band Gap (Symbolic Regression Equations)
    for target in ['Total Magnetization (uB)', 'Energy Above Hull (eV)', 'Band Gap (eV)']:
        sym_res = symbolic_results[target]
        target_short = 'Total Magnetization (M)' if 'Magnetization' in target else ('Energy Above Hull' if 'Hull' in target else 'Band Gap (Eg)')
        print(f"{target_short:<30} | {sym_res['R2']:.4f} | {'Symbolic Regression':<20} | {sym_res['Equation']:<45}")
    print("="*110 + "\n")
    
    print("Pipeline executed successfully. All output figures are saved under the Results/ folder.")

if __name__ == "__main__":
    main()
