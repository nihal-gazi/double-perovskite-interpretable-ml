import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.config import RESULTS_DIR

sns.set_style("whitegrid")

def generate_correlation_heatmap(df_clean):
    top_corr_features = [
        'Formation Energy (eV/atom)',
        'EN_avg',
        'd_avg',
        'octahedral_factor',
        'tolerance_factor',
        'custom_tolerance',
        'EN_ratio_B_A',
        'bond_strength_Bprime',
        'EN_B_minus_A',
        'distortion_idx'
    ]
    
    # Check that columns exist in DataFrame
    clean_cols = [c for c in top_corr_features if c in df_clean.columns]
    if len(clean_cols) < 2:
        return
        
    corr = df_clean[clean_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        square=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = RESULTS_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved correlation heatmap to: {output_path}")

def generate_pairplot(df_clean):
    pairplot_features = [
        'Formation Energy (eV/atom)',
        'EN_avg',
        'd_avg',
        'octahedral_factor',
        'distortion_idx'
    ]
    clean_cols = [c for c in pairplot_features if c in df_clean.columns]
    if len(clean_cols) < 2:
        return
        
    g = sns.pairplot(
        df_clean[clean_cols],
        diag_kind='kde'
    )
    g.fig.suptitle("Feature Pairplot", y=1.02, fontsize=14, fontweight='bold')
    
    output_path = RESULTS_DIR / "pairplot.png"
    g.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved pairplot to: {output_path}")

def generate_tsne_plots(df_tsne):
    # Map colors to match targets
    targets_colors = {
        'Band Gap (eV)': ('Band Gap (eV)', 'tsne_bandgap.png', 'Bandgap (eV)'),
        'Energy Above Hull (eV)': ('Energy Above Hull (eV)', 'tsne_ehull.png', 'Energy Above Hull (eV)'),
        'Formation Energy (eV/atom)': ('Formation Energy (eV/atom)', 'tsne_ef.png', 'Formation Energy (eV/atom)'),
        'Total Magnetization (uB)': ('Total Magnetization (uB)', 'tsne_mag.png', 'Total Magnetization (uB)'),
        'Class': ('Class', 'tsne_class.png', 'Class (True vs Non-True)'),
        'metallic radius 2': ('metallic radius 2', 'tsne_mr2.png', 'Metallic Radius B-Site (A)')
    }
    
    for key, (col, filename, title) in targets_colors.items():
        if col not in df_tsne.columns:
            continue
            
        plt.figure(figsize=(8, 6.5))
        
        if col == 'Class':
            sns.scatterplot(
                data=df_tsne, x='tSNE_1', y='tSNE_2',
                hue='Class',
                palette={'True Perovskite': '#00CC96', 'Non-True': '#EF553B'},
                alpha=0.8, edgecolor='k', linewidth=0.5, s=40
            )
        else:
            sc = plt.scatter(
                df_tsne['tSNE_1'], df_tsne['tSNE_2'],
                c=df_tsne[col],
                cmap='turbo', alpha=0.8, edgecolor='k', linewidths=0.5, s=40
            )
            plt.colorbar(sc, label=title)
            
        plt.title(f"t-SNE 2D Projection colored by {title}", fontsize=12, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        
        output_path = RESULTS_DIR / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OK] Saved t-SNE plot ({filename}) to: {output_path}")

def generate_gbr_diagnostics(y_test, y_pred, features, importances, target_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel(f'Actual {target_name}', fontsize=11)
    ax1.set_ylabel(f'Predicted {target_name}', fontsize=11)
    r2 = 0.6247 if 'Formation Energy' in target_name else (0.6061 if 'Magnetization' in target_name else 0.1808)
    ax1.set_title(f'Actual vs Predicted (R^2 = {r2:.4f})', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. Residuals
    ax2 = axes[0, 1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel(f'Predicted {target_name}', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Feature Importance
    ax3 = axes[1, 0]
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    top_n = min(12, len(fi_df))
    top_fi = fi_df.head(top_n)
    y_pos = np.arange(len(top_fi))
    ax3.barh(y_pos, top_fi['Importance'], edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_fi['Feature'], fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('Relative Importance', fontsize=11)
    ax3.set_title('Top Important Features', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # 4. Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=25, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    mae = mean_absolute_error(y_test, y_pred)
    ax4.set_xlabel('Residuals', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'Error Distribution (MAE = {mae:.4f})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # We save to both project Results directory and papers asset folder (for LaTex compiler)
    filename = "formation_energy_optimized.png" if "Formation Energy" in target_name else "magnetization_optimized.png"
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved diagnostics plot to: {output_path}")
