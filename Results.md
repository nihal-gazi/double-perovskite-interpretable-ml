# VAE + tSNE
## Results on Dim=64
So, it seems that, we have some interesting results.
The error logs are
```
Training on: cuda
Epoch 100/1000 Complete. Loss = 93.82587432861328
Epoch 200/1000 Complete. Loss = 75.61445617675781
Epoch 300/1000 Complete. Loss = 81.31501007080078
Epoch 400/1000 Complete. Loss = 83.59830474853516
Epoch 500/1000 Complete. Loss = 85.49809265136719
Epoch 600/1000 Complete. Loss = 71.2939453125
Epoch 700/1000 Complete. Loss = 77.97398376464844
Epoch 800/1000 Complete. Loss = 93.74501037597656
Epoch 900/1000 Complete. Loss = 78.81514739990234
Epoch 1000/1000 Complete. Loss = 82.0006103515625
```

Obviously, there are 8 properties and it is not rational to put 8 properties through a 64 dimension embedding.
The loss did good amount of oscillation.

## Results on Dim=3

Here the loss looked like:
```
Training on: cuda
Epoch 100/1000 Complete. Loss = 83.13053894042969
Epoch 200/1000 Complete. Loss = 85.798095703125
Epoch 300/1000 Complete. Loss = 86.87324523925781
Epoch 400/1000 Complete. Loss = 99.27943420410156
Epoch 500/1000 Complete. Loss = 92.72110748291016
Epoch 600/1000 Complete. Loss = 75.08440399169922
Epoch 700/1000 Complete. Loss = 94.97413635253906
Epoch 800/1000 Complete. Loss = 84.98504638671875
Epoch 900/1000 Complete. Loss = 90.20970153808594
Epoch 1000/1000 Complete. Loss = 80.8459701538086
```

Oscillation is present here, but the network seemed to have stumbled accross a slightly better loss than what it started (83 -> 80)

### tSNE 2D Output:

||||
| :---: | :---: | :---: |
| ![Bandgap](Results/3_2d_bandgap.png) Bandgap| ![Class](Results/3_2d_class.png) Class| ![Energy Above Hull](Results/3_2d_energy_above_hull.png) Energy above hull|
| ![Formation Energy](Results/3_2d_formation_energy.png) Formation Energy| ![MR2](Results/3_2d_mr2.png) Metallic Radius 2| ![MR4](Results/3_2d_mr4.png) Metallic Radius 4|
| ![Total Mag](Results/3_2d_total_mag.png) Total Magnetization| | |

(Metallic Radius 1 and 3 were omitted, because they were not very interesting)

Conclusion: Except "Class", all had very interesting understandable trends - hence they were chosen as learnable properties


# Symbolic Regression

We ran a Symbolic Regression on all learnable properties.
Additional Data we gave:
1. EN of A, B, O
2. All Bond Lengths

Long story short:
1. Formation Energy(eV/atom) seemed to have the best $R^2$.
2. Others had negative $R^2$ indicating we could not find any pattern in the data.

Symbolic Regression details:
- Uses Genetic Algorithm
- Had only $+$, $-$, $\times$, $\div$, $\sqrt{x}$, $|x|$ operations to work with

So, input features for the Symbolic Regression:
- $\mathrm{EN}_A$ : Electronegativity of the A-site atom  
- $\mathrm{EN}_B$ : Electronegativity of the B-site atom  
- $\mathrm{EN}_{B'}$ : Electronegativity of the B′-site atom  
- $\mathrm{EN}_O$ : Electronegativity of oxygen  
- $d_{A-O}$ : Bond length between A and O atoms  
- $d_{B-O}$ : Bond length between B and O atoms  
- $d_{B'-O}$ : Bond length between B′ and O atoms  

(Where the double pervoskite is $A_2BB'O_6$)

<br><br>

Results from `Analysis_v1/Symbolic_Regression/symbolic_regression_v2.py` (This did a `generation=40` Symbolic Regression on all learnable properties):
```
Fetching Electronegativity values from Mendeleev database...

==================================================
SYMBOLIC REGRESSION: ELECTRONIC & STRUCTURAL DISCOVERY
==================================================

🎯 Target Property: Total Magnetization (Î¼B)
📊 R² Accuracy: 0.137
📐 Discovered Equation:
Total Magnetization (Î¼B) ≈ add(add(abs(mul(sub(sqrt(abs(abs([bond_length_A_O]))), add([EN_A], [EN_A])), add(abs(sqrt(sqrt(add([EN_O], sub([bond_length_B_O], 0.507))))), sub(abs([bond_length_A_O]), sub(sub(mul([bond_length_B_O], [EN_A]), add([EN_A], [EN_A])), [EN_B]))))), sub(abs([EN_O]), div([bond_length_B_O], [EN_B]))), add(div(add(div(mul(sub([EN_B], [bond_length_B_O]), sub([bond_length_Bprime_O], 0.424)), div(div([EN_O], [bond_length_A_O]), sqrt([EN_B]))), add(sqrt(sqrt(div([EN_O], sub(div(0.044, [EN_O]), [EN_A])))), div(sqrt([bond_length_A_O]), mul([bond_length_A_O], [bond_length_B_O])))), div(sqrt([EN_A]), sqrt([EN_B]))), add([EN_A], [EN_Bprime])))

🎯 Target Property: Formation Energy (eV/atom)
📊 R² Accuracy: 0.274
📐 Discovered Equation:
Formation Energy (eV/atom) ≈ sub(sub([EN_B], [bond_length_Bprime_O]), sub(-0.869, sub(mul(0.382, [EN_Bprime]), mul([EN_O], 0.955))))

🎯 Target Property: Energy Above Hull (eV)
📊 R² Accuracy: -0.010
📐 Discovered Equation:
Energy Above Hull (eV) ≈ 0.074

🎯 Target Property: Band Gap (eV)
📊 R² Accuracy: -0.284
📐 Discovered Equation:
Band Gap (eV) ≈ div([EN_A], [bond_length_A_O])
```

The total Magnetization equation that it found, if rewritten in a easy-to-understand form, becomes:
$$
\begin{aligned}
T_1 =\;&
\left|
\left(
\sqrt{\left|\left|\text{bond\_length}_{A-O}\right|\right|}
-(EN_A+EN_A)
\right)
\left(
\left|\sqrt{\sqrt{EN_O+(\text{bond\_length}_{B-O}-0.507)}}\right|
-\left|
\text{bond\_length}_{A-O}
-\Big(
(\text{bond\_length}_{B-O}\cdot EN_A)
-(EN_A+EN_A)
-EN_B
\Big)
\right|
\right)
\right|
\\[6pt]
T_2 =\;&
\left|EN_O\right|
-\frac{\text{bond\_length}_{B-O}}{EN_B}
\\[6pt]
T_3 =\;&
\frac{
\operatorname{add}\Bigg(
\frac{
\big(EN_B-\text{bond\_length}_{B-O}\big)
\big(\text{bond\_length}_{B'-O}-0.424\big)
}{
\left(\frac{EN_O}{\text{bond\_length}_{A-O}}\right)\big/\sqrt{EN_B}
},
\;
\sqrt{\sqrt{\frac{EN_O}{\left(\frac{0.044}{EN_O}\right)-EN_A}}}
+
\frac{\sqrt{\text{bond\_length}_{A-O}}}{\text{bond\_length}_{A-O}\cdot\text{bond\_length}_{B-O}}
\Bigg)
}{
\sqrt{EN_A}/\sqrt{EN_B}
}
\\[8pt]
\text{Total Magnetization }(\mu_B) \approx\;&
\operatorname{add}\Big(
\operatorname{add}(T_1, T_2),
\operatorname{add}(T_3, (EN_A+EN_{B'}))
\Big)
\end{aligned}
$$

This is totall gibberish.
It overfit the noisy and as we can see, there are lots of micro-adjustment terms(a indicative of the $R^2 = 0.137$) in the equation - hence it is safe to let go of.

But, the formation energy gave a much more interesting & meaningful equation($R^2 = 0.274$)

$$
\begin{aligned}
\text{Formation Energy (eV/atom)}
\approx\;&
\Big(
EN_B
-
\text{bond\_length}_{B'-O}
\Big)
\\[6pt]
&-
\Big(
-0.869
-
\big(
0.382 \cdot EN_{B'}
\big)
+
\big(
0.955 \cdot EN_O
\big)
\Big)
\end{aligned}
$$

A further deep-dive test was run on Formation Energy, and we found that, upon running Symbolic Regression for `generation=1000`, it literally **invented it's own novel thermodynamic equation** - it was ***brilliant!***

Results from `Analysis_v1/Symbolic_Regression/symbolic_regression_v3_formation_energy.py` (This did a `generation=1000` Symbolic Regression on all learnable properties):
```
Fetching Electronegativity values from Mendeleev database...

==================================================
SYMBOLIC REGRESSION: ELECTRONIC & STRUCTURAL DISCOVERY
==================================================

🎯 Target Property: Formation Energy (eV/atom)
Evolving Formation Energy (eV/atom): 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [22:10<00:00,  1.33s/gen]

📊 R² Accuracy: 0.385
📐 Discovered Equation:
Formation Energy (eV/atom) ≈ sub(sub([EN_B], [bond_length_Bprime_O]), sub(-0.892, sub(mul(0.382, div(div([EN_Bprime], [EN_A]), div([bond_length_B_O], [bond_length_Bprime_O]))), mul([EN_O], 0.955))))
```

The Formation Energy formula it found - was extraordinarily simple yet profound:
$$
\begin{aligned}
\text{Formation Energy (eV/atom)}
(E_f)
\approx\;&
\Big(
EN_B
-
\text{bond\_length}_{B'-O}
\Big)
\\[6pt]
&-
\Bigg(
-0.892
-
\Bigg(
0.382 \cdot
\frac{
\left(
\frac{EN_{B'}}{EN_A}
\right)
}{
\left(
\frac{\text{bond\_length}_{B-O}}
{\text{bond\_length}_{B'-O}}
\right)
}
\Bigg)
+
(0.955 \cdot EN_O)
\Bigg)
\end{aligned}
$$



$$
\implies E_f \approx EN_B - d_{B'O} + 0.892 + 0.382\frac{EN_{B'}d_{B'O}}{EN_A d_{BO}} - 0.955 EN_O
$$

The AI literally invented a **Custom Tolerance Factor**!

- The numerator ($EN_{B'} / EN_A$) calculates the **electronic contrast** (how differently the B' site and A site attract electrons).
- The denominator ($d_{B-O} / d_{B'-O}$) calculates the **geometric distortion** (how structurally different the two transition metal octahedra are).

The AI figured out that thermodynamic stability (Formation Energy) is a direct ratio of *charge distribution* divided by *geometric strain*. It successfully merged quantum chemistry (Electronegativity) with crystallography (Bond Lengths) into a single, elegant term.



# GB Trees + Symbolic Regression

### Methodology

To predict the **Formation Energy** ($\Delta E_f$) of double perovskites while maintaining human interpretability, we employed a two-step hybrid machine learning approach:

1. **Feature Discovery (Gradient Boosting):** We engineered domain-specific chemical descriptors (e.g., Shannon radii, valence electrons, Goldschmidt tolerance factor) and trained a Gradient Boosting Regressor. This tree-based model identified the most critical physical drivers of stability.
2. **Mathematical Distillation (Symbolic Regression):** To break open the "black box" of the GB model, we restricted a Symbolic Regressor to use *only* the top features identified by the tree model, forcing the AI to derive a clean, factorable algebraic equation.

### Model Performance

* **Gradient Boosting (Black Box):** $R^2 = 0.6247$ | MAE = 0.1371 eV/atom
* *Top Features Identified:* Average Electronegativity ($EN_{avg}$), Average B-O Bond Length ($d_{avg}$), Octahedral Factor ($\mu_{oct}$), and Tolerance Factor ($t$).


* **Symbolic Regression (White Box Equation):** $R^2 = 0.533$

### Final Discovered Formula

The Genetic Programming algorithm distilled the thermodynamic stability down to the following equation:

$$\Delta E_{f} \approx t \cdot (EN_{avg} - \mu_{oct}) - d_{avg} - 0.913$$

**Physical Interpretation:**
The AI independently derived that the thermodynamic stability of a double perovskite is governed by:

1. **Electronic-Structural Coupling:** The structural stability (Tolerance Factor, $t$) is directly coupled with the bond strength (Average Electronegativity, $EN_{avg}$).
2. **Geometric Strain Penalties:** Increases in local distortion (Octahedral Factor, $\mu_{oct}$) or the stretching of the lattice (Average Bond Length, $d_{avg}$) act as energetic penalties, reducing the overall stability of the crystal.


# Deep GNN Features (CHGNet) + GB Trees & Symbolic Regression

### Legend
The following descriptors are used in the equations and models below:
- **GNN Quantum Features**:
  - $E_{GNN}$ (or `CHGNet_Energy`): Potential energy per atom of the relaxed structure predicted by CHGNet (eV/atom).
  - $M_{net}$ (or `CHGNet_Net_Magmom`): Net magnetic moment of the unit cell ($\mu_B$).
  - $M_{abs}$ (or `CHGNet_Abs_Magmom`): Sum of absolute values of local magnetic moments ($\mu_B$), capturing local unpaired spins.
- **Crystallographic & Chemical Descriptors**:
  - $t$ (or `Tolerance_Factor`): Goldschmidt tolerance factor.
  - $\mu_{oct}$ (or `Octahedral_Mismatch`): Octahedral mismatch/distortion index between B and B' sites.
  - $EN_{avg}$ (or `EN_avg`): Average electronegativity of B and B' site transition metals.
  - $EN_A$ (or `EN_A`): Electronegativity of the A-site cation.
  - $Shannon_A$ (or `Shannon_A`): Shannon ionic radius of the A-site element (Å).
  - $Shannon_B$ (or `Shannon_B`): Shannon ionic radius of the B-site element (Å).
  - $Val_{avg}$ (or `Val_avg`): Average valence electron count of the transition metals.

### Methodology
To predict electronic, magnetic, and hull stability properties of double perovskites ($A_2BB'O_6$), we integrated pre-trained Deep Graph Neural Network (GNN) descriptors from the **CHGNet** framework (Deng et al., *Nature Machine Intelligence*, 2023) with our traditional chemical descriptors.
We extracted the following quantum parameters from the relaxed structures:
- **Relaxed Energy per Atom ($E_{GNN}$)**: Energetic state calculated by the GNN (eV/atom).
- **Net Magnetization ($M_{net}$)**: The cell's absolute net magnetic moment ($\mu_B$).
- **Absolute Magnetization ($M_{abs}$)**: Total local spin moments summed over all atoms ($\mu_B$), capturing local unpaired spins even in antiferromagnetic alignments.

We then benchmarked model performance using two approaches:
1. **Gradient Boosting Trees**: Tree-based black-box modeling to establish an upper performance limit and extract feature importances.
2. **Symbolic Regression (Genetic Programming)**: White-box formula discovery to find clean, factorable algebraic expressions.

### Citation
> Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modeling. *Nature Machine Intelligence* 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3

### Performance Summary & Feature Importance

| Target Property | Model | $R^2$ Score | MAE | Top Drivers & Importances |
| :--- | :--- | :---: | :---: | :--- |
| **Total Magnetization ($μ_B$)** | Gradient Boosting <br> Symbolic Regression | **0.6061** <br> **0.6281** | 2.8001 <br> 2.9258 | `CHGNet_Net_Magmom` ($39.5\%$), `CHGNet_Abs_Magmom` ($29.8\%$), `CHGNet_Energy` ($19.1\%$), `Tolerance_Factor` ($5.9\%$) |
| **Band Gap (eV)** | Gradient Boosting <br> Symbolic Regression | **0.1808** <br> -0.1467 | 0.6438 <br> 0.7850 | `CHGNet_Energy` ($33.8\%$), `EN_avg` ($18.2\%$), `CHGNet_Net_Magmom` ($16.7\%$), `Tolerance_Factor` ($12.4\%$) |
| **Energy Above Hull (eV)** | Gradient Boosting <br> Symbolic Regression | **0.3424** <br> **0.3985** | 0.0400 <br> 0.0442 | `CHGNet_Energy` ($59.4\%$), `Tolerance_Factor` ($9.7\%$), `CHGNet_Net_Magmom` ($9.0\%$), `CHGNet_Abs_Magmom` ($8.8\%$) |

---

### Discovered Formulas

#### 1. Total Magnetization ($M$)
The Genetic Programming regressor successfully matched the Gradient Boosting performance ($R^2 = 0.6281$) with the following discovered relation:

$$M \approx \left| \frac{t}{E_{GNN}} + E_{GNN} + EN_{avg} + \left| \frac{EN_{avg}}{E_{GNN}} + E_{GNN} + 2 EN_{avg} + M_{abs} - 0.249 \right| \right|$$

*Raw Program:*
`Total Magnetization (uB) = abs(abs(add(add(div(abs([Tolerance_Factor]), [CHGNet_Energy]), add([CHGNet_Energy], [EN_avg])), abs(abs(abs(add(add(div([EN_avg], [CHGNet_Energy]), add([CHGNet_Energy], [EN_avg])), sub([CHGNet_Abs_Magmom], add(sub(0.748, [EN_avg]), -0.499)))))))))`

#### 2. Band Gap ($E_g$)
The symbolic regressor discovered the following formulation, though the negative test $R^2$ score ($-0.1467$) indicates it does not generalize well to unseen validation materials:

$$E_g \approx E_{GNN} + 0.980$$

*Raw Program:*
`Band Gap (eV) = add(0.980, [CHGNet_Energy])`

#### 3. Energy Above Hull ($E_{hull}$)
The symbolic regressor discovered a highly clean, linear scaling relationship with the GNN-relaxation energy ($R^2 = 0.3985$):

$$E_{hull} \approx -0.142 \cdot E_{GNN}$$

*Raw Program:*
`Energy Above Hull (eV) = mul([CHGNet_Energy], -0.142)`

**Physical Interpretation:**
1. **Total Magnetization Coupling**: The formula demonstrates coupling between structural factors ($t$) and quantum spin configurations ($M_{abs}$).
2. **thermodynamic convex hull stability**: Stability above the thermodynamic hull is directly proportional to the relaxation energy of the crystal lattice calculated by the GNN potential. The negative scaling factor indicates that lower energy states (more negative $E_{GNN}$) correspond to larger stabilization barriers, as expected.


# Linear Regression Benchmarks (All Gathered Data)

We ran standardized linear regression models (Ordinary Least Squares, Ridge, and Lasso) using all traditional crystallographic features, true 3D spatial properties (volume and density), and CHGNet GNN quantum features. The models were optimized on train-test splits (80/20) and compared to establish simple linear baseline equations.

### Performance Summary

| Target Property | Best Linear Model | Test $R^2$ Score | Test MAE | Top Significant Features |
| :--- | :--- | :---: | :---: | :--- |
| **Formation Energy (eV/atom)** | Lasso Regression (CV) | **0.6263** | 0.1778 | `Val_avg`, `EN_A`, `EN_avg`, `Shannon_A` |
| **Total Magnetization ($μ_B$)** | OLS Linear Regression | **0.6019** | 2.7835 | `Shannon_A`, `Tolerance_Factor`, `CHGNet_Net_Magmom` |
| **Band Gap (eV)** | Lasso Regression (CV) | -0.2090 | 0.7685 | `CHGNet_Energy`, `Val_B`, `Val_Bprime` |
| **Energy Above Hull (eV)** | OLS Linear Regression | **0.3610** | 0.0464 | `Tolerance_Factor`, `Shannon_A`, `CHGNet_Energy` |

---

### Discovered Linear Equations

#### 1. Formation Energy ($\Delta E_{f}$)
$$\Delta E_f \approx 0.1508 \cdot Val_{avg} - 1.0197 \cdot EN_A + 0.4113 \cdot EN_{avg} - 0.2725 \cdot Shannon_A - 2.6972$$

#### 2. Total Magnetization ($M$)
$$M \approx -34.3562 \cdot Shannon_A + 84.1543 \cdot Tolerance\_Factor + 0.3864 \cdot M_{net} + 0.3864 \cdot M_{abs} - 68.0119$$

#### 3. Energy Above Hull ($E_{hull}$)
$$E_{hull} \approx 2.2146 \cdot Tolerance\_Factor - 0.7490 \cdot Shannon_A - 0.1614 \cdot E_{GNN} + 0.4777 \cdot Shannon_B - 1.5616$$