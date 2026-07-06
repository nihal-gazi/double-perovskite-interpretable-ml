# Final Results & Analysis

This document synthesizes the best-performing models, discovered physical equations, and key realizations derived from the Double Perovskite Machine Learning Pipeline.

> **Dataset Scope Note:**
> * **Exploratory Visualizations (VAE & t-SNE)**: Analyzed on the **combined dataset** (including both *true* and *non-true* double perovskite compounds) to map the global material representation.
> * **Predictive Regressions & Algebraic Discoveries**: Performed **exclusively on true double perovskites** (derived from the *approx true double perovskite* sheet) to model only synthesizable structures.

---

## 🏆 Best Performing Models & Discovered Equations

### 1. Formation Energy ($\Delta E_f$) *(Dataset: True Only)*
- **Highest Performing Formula**: Lasso Linear Regression ($R^2 = 0.6263$)
- **Gradient Boosting Upper Limit**: $R^2 = 0.6247$
- **Discovered Equation (Lasso Regression)**:
  $$\boxed{\Delta E_f \approx 0.1508 \cdot Val_{avg} - 1.0197 \cdot EN_A + 0.4113 \cdot EN_{avg} - 0.2725 \cdot Shannon_A - 2.6972}$$
  
  *Alternative Symbolic Discoveries:*
  - Distilled GB Tree Logic ($R^2 = 0.533$):
    $$\Delta E_{f} \approx t \cdot (EN_{avg} - \mu_{oct}) - d_{avg} - 0.913$$
  - Custom Tolerance Factor Breakthrough ($R^2 = 0.385$):
    $$E_f \approx EN_B - d_{B'O} + 0.892 + 0.382\frac{EN_{B'}d_{B'O}}{EN_A d_{BO}} - 0.955 EN_O$$

### 2. Total Magnetization ($M$) *(Dataset: True Only)*
- **Highest Performing Formula**: Symbolic Regression ($R^2 = 0.6281$)
- **Gradient Boosting Upper Limit**: $R^2 = 0.6061$
- **Discovered Equation (Symbolic)**: 
  $$\boxed{M \approx \left| \frac{t}{E_{GNN}} + E_{GNN} + EN_{avg} + \left| \frac{EN_{avg}}{E_{GNN}} + E_{GNN} + 2 EN_{avg} + M_{abs} - 0.249 \right| \right|}$$
  *Demonstrates complex coupling between structural parameters (Tolerance Factor $t$) and quantum spin parameters.*

### 3. Energy Above Hull ($E_{hull}$) *(Dataset: True Only)*
- **Highest Performing Formula**: Symbolic Regression ($R^2 = 0.3985$)
- **Best Linear Baseline**: OLS Linear Regression ($R^2 = 0.3610$)
- **Discovered Equation (Symbolic)**: 
  $$\boxed{E_{hull} \approx -0.142 \cdot E_{GNN}}$$
  *Revealed a remarkably clean, linear scaling relationship with the GNN-relaxation energy.*

### 4. Band Gap ($E_g$) *(Dataset: True Only)*
- **Highest Performing Model**: Gradient Boosting ($R^2 = 0.1808$) *(No analytical algebraic equation)*
- **Best Analytical Equation**: Symbolic Regression ($R^2 = -0.1467$):
  $$\boxed{E_g \approx E_{GNN} + 0.980}$$
  *(Note: Negative $R^2$ indicates this model does not generalize well to unseen test validation materials).*

---

## 📋 Variables Legend

| Variable Symbol | Full Description | Physical Meaning | Sourcing & Extraction Method |
| :--- | :--- | :--- | :--- |
| **$t$** | Goldschmidt Tolerance Factor | Geometric measure of the structural stability and distortion of the perovskite crystal lattice. | Calculated geometrically from Shannon ionic radii: $t = \frac{r_A + r_O}{\sqrt{2}(r_{avg} + r_O)}$, where ionic radii are retrieved from the **Mendeleev database** and cell topology is parsed using **PyMatgen** from Materials Project `.cif` files. |
| **$EN_{avg}$** | Average Electronegativity | Average electronegativity (Pauling scale) of the transition metals occupying the B and B' sites. | Sourced from the **Mendeleev database** using the chemical symbols of B and B' site cations. |
| **$\mu_{oct}$** | Octahedral Mismatch | Mismatch/distortion index between the sizes of the B and B' site octahedra ($\mu_{oct} = r_B / r_{B'}$). | Calculated using Shannon ionic radii from the **Mendeleev database** during 3D structural analysis. |
| **$d_{avg}$** | Average Bond Length | The average length of B-O and B'-O bonds in the unit cell (in Å). | Calculated by computing neighbor distances from `.cif` coordinate structures (downloaded from the **Materials Project REST API**) via **PyMatgen**. |
| **$EN_A$ / $EN_B$ / $EN_{B'}$ / $EN_O$** | Elemental Electronegativities | Electronegativity values for the respective atoms in the $A_2BB'O_6$ lattice. | Sourced from the **Mendeleev database** (Pauling Electronegativity scale). |
| **$d_{BO}$ / $d_{B'O}$** | Specific Octahedral Bond Lengths | Individual bond lengths (in Å) between the transition metal cations and coordinating oxygen anions. | Derived from structural cell coordinates in Materials Project `.cif` files using **PyMatgen**. |
| **$E_{GNN}$** | CHGNet Relaxed Potential Energy | Relaxed total energy per atom (eV/atom) predicting lattice structural stability. | Computed using the pretrained **CHGNet (v0.3.0)** Graph Neural Network potential (developed by Berkeley/Materials Project) on the double perovskite crystal structures. |
| **$M_{abs}$** | CHGNet Absolute Magnetization | Sum of absolute values of local magnetic spin moments ($\mu_B$), capturing local unpaired spins. | Computed using the pretrained **CHGNet (v0.3.0)** model by summing absolute site-specific magnetic moments. |
| **$Val_{avg}$** | Average Valence | Average number of valence electrons of the B and B' site cations, indicating bonding capacity. | Extracted from the **Mendeleev database** using the chemical symbols of B and B' site cations. |
| **$Shannon_A$** | Shannon Ionic Radius of A-site | The effective ionic radius (in Å) of the A-site cation in its specific coordination environment. | Sourced from the **Mendeleev database** using the A-site elemental symbol. |

---

## 💡 Key Realizations

1. **AI-Driven Physics Discovery**: The Genetic Programming algorithm proved highly capable of distilling "black box" tree-based models into "white box" interpretable equations. In the case of Formation Energy, it successfully merged quantum chemistry (Electronegativity) with crystallography (Bond Lengths) to create a custom geometric-electronic Tolerance Factor.
2. **The Power of the Quantum Oracle**: Utilizing a pre-trained Graph Neural Network (CHGNet) as an "Oracle" to extract latent quantum properties (relaxed energies, magnetic spin configurations) proved highly effective. These extracted $E_{GNN}$ and $M_{abs}$ features consistently arose as top drivers in tree-based models and symbolic regressions for predicting thermodynamic and magnetic target variables.
3. **Electronic-Structural Coupling**: Both the feature importance analyses and the discovered equations repeatedly highlighted that stability and magnetism in double perovskites are not driven by single parameters, but by the explicit coupling of geometric strain (e.g., octahedral mismatch, tolerance factor) with electronic properties (e.g., electronegativity).

---

## 🚧 Limitations & Constraints Reached

1. **The Band Gap Barrier**: Predicting the Band Gap proved exceptionally difficult. All tested models (Linear, Tree-based, Symbolic) largely failed to predict this property accurately ($R^2_{max} \approx 0.18$). This indicates that our current set of 1D chemical descriptors and 3D geometric/quantum properties are insufficient. The Band Gap likely requires explicit orbital configurations or many-body interaction mappings that simple structural parameters cannot capture.
2. **Accuracy Ceiling**: Across the most successful target properties (Formation Energy, Total Magnetization), performance maxed out around $R^2 \approx 0.62 - 0.63$. This ceiling suggests an inherent limit to the predictive power of the current dataset. 
3. **Symbolic Overfitting**: While Genetic Programming can find elegant equations, it is highly prone to overfitting noise. Without strict regularization or by allowing the generation count to run too long without proper constraints, the AI can produce highly convoluted "gibberish" equations (as seen in early Magnetization attempts) that fail to map to true physical phenomena.
