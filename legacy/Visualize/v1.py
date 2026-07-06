# ============================================================
# ADVANCED VISUALIZATION SUITE
# ============================================================

print(f"\n{'='*80}")
print("ADVANCED DATA VISUALIZATION")
print(f"{'='*80}\n")

sns.set_style("whitegrid")

# ------------------------------------------------------------
# 1. Correlation Heatmap
# ------------------------------------------------------------

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

corr = df_clean[top_corr_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    square=True
)

plt.title("Feature Correlation Heatmap", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 2. Pairplot
# ------------------------------------------------------------

pairplot_features = [
    'Formation Energy (eV/atom)',
    'EN_avg',
    'd_avg',
    'octahedral_factor',
    'distortion_idx'
]

sns.pairplot(
    df_clean[pairplot_features],
    diag_kind='kde'
)

plt.savefig(OUTPUT_DIR / "pairplot.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 3. Individual Scatter Plots
# ------------------------------------------------------------

important_features = [
    'EN_avg',
    'd_avg',
    'octahedral_factor',
    'tolerance_factor',
    'distortion_idx',
    'custom_tolerance'
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, feature in zip(axes.flatten(), important_features):

    ax.scatter(
        df_clean[feature],
        df_clean[target],
        alpha=0.6,
        s=40
    )

    # Trend line
    z = np.polyfit(df_clean[feature], df_clean[target], 1)
    p = np.poly1d(z)

    x_vals = np.linspace(
        df_clean[feature].min(),
        df_clean[feature].max(),
        100
    )

    ax.plot(x_vals, p(x_vals), linewidth=2)

    ax.set_xlabel(feature)
    ax.set_ylabel("Formation Energy")
    ax.set_title(feature)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_scatterplots.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 4. Feature Importance Plot
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

top_n = 15
top_fi = fi_df.head(top_n)

sns.barplot(
    data=top_fi,
    x=col_name,
    y='Feature'
)

plt.title("Top Feature Importances", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_plot.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 5. Residual Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

sns.histplot(
    residuals,
    kde=True,
    bins=30
)

plt.axvline(0, linestyle='--')

plt.title("Residual Distribution", fontsize=15, fontweight='bold')
plt.xlabel("Prediction Error")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 6. Actual vs Predicted
# ------------------------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    best_pred,
    alpha=0.7,
    s=50
)

min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--',
    linewidth=2
)

plt.xlabel("Actual Formation Energy")
plt.ylabel("Predicted Formation Energy")

plt.title(
    f"Actual vs Predicted\nR² = {best_r2:.4f}",
    fontsize=15,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 7. Nonlinear Trend Exploration
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

sns.regplot(
    x=df_clean['EN_avg'],
    y=df_clean[target],
    lowess=True,
    scatter_kws={'alpha':0.5}
)

plt.title(
    "Nonlinear Trend: EN_avg vs Formation Energy",
    fontsize=15,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nonlinear_trend_EN_avg.png", dpi=300)
plt.show()

print("\n✓ All visualization plots saved.")