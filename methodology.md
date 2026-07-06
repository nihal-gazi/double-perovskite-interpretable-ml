# Project Methodology

This document explains our step-by-step scientific workflow for predicting double perovskite ($A_2BB'O_6$) stability, magnetism, and band gaps, and for discovering new physical equations. 

To bridge the gap between materials science and machine learning, we structured the pipeline to move systematically from **raw crystal structures** to **interpretable physical laws**.

---

## 🛠️ Overview of Machine Learning Tools

Below is a reference guide explaining what each algorithm does and how it fits into our physical pipeline.

| Tool / Algorithm | What It Does (In Simple Terms) | Role in Our Pipeline |
| :--- | :--- | :--- |
| **CHGNet (Graph Neural Network)** | An AI "Quantum Oracle" trained to simulate atomic forces and electron charge distributions. | Acts as an ultra-fast alternative to Density Functional Theory (DFT), providing relaxed ground-state energies and spin states directly from crystal structures. |
| **Variational Autoencoder (VAE)** | A compression tool that maps a high-dimensional property space down to a 3D coordinate system. | Projects the complex landscape of perovskite properties so we can check if materials group together geometrically. |
| **t-SNE (t-Distributed Stochastic Neighbor Embedding)** | A visualization tool that flattens multi-dimensional structures into a 2D map while preserving local neighborhoods. | Let us visually verify if target properties (like magnetism or band gaps) display smooth, learnable gradients or random noise. |
| **Linear Regression (OLS, Ridge, Lasso)** | Fitting a straight-line equation ($y = m_1x_1 + m_2x_2 + ... + c$). Lasso regression automatically forces weak coefficients to zero. | Establishes our baseline physical model and filters out unimportant chemical descriptors. |
| **Gradient Boosting Trees** | A model that makes predictions using a sequence of simple "yes/no" decision paths, where each tree corrects the mistakes of the last. | Tells us the absolute performance limit of the dataset and ranks which physical descriptors (e.g., radius, electronegativity) are the most critical. |
| **Symbolic Regression (Genetic Programming)** | An evolutionary search that tests, mutates, and combines algebraic operators ($+, -, \times, \div$) to fit the data. | Discovers human-readable physical equations by building formulas out of the top physical descriptors. |

---

## 🚀 The 4-Step Pipeline Flow

Our research followed a logical progression: we gathered physical descriptors, verified their distribution, tested the limits of predictability, and distilled the results into clean algebraic laws.

```text
[Step 1: Data & Quantum Simulation]
                │
                ▼ (1D Chemistry & 3D Coordinates)
                │
[Step 2: Visual Trend Analysis]
                │
                ▼ (Verified Predictable Properties)
                │
[Step 3: Finding Predictability Bounds]
                │
                ▼ (Top Physical Drivers & Feature Importances)
                │
[Step 4: Algebraic Equation Discovery]
```

### Step 1: Gathering Physical & Quantum Descriptors
To understand each material, we combined three different levels of physical information:
1. **Chemical Properties**: Sourced atomic electronegativities ($EN$) and Shannon ionic radii from the **Mendeleev database**.
2. **Geometric Properties**: Downloaded 3D crystal structure files (`.cif`) from the **Materials Project REST API** and calculated unit cell volumes, densities, Goldschmidt Tolerance Factors ($t$), and Octahedral Mismatches ($\mu_{oct}$) via the **PyMatgen** library.
3. **Quantum States**: Passed the structures through the pretrained **CHGNet** network to obtain the relaxed atomic energies ($E_{GNN}$) and magnetic spin configurations ($M_{net}, M_{abs}$).

### Step 2: Visualizing Trends (VAE & t-SNE)

Before building predictive models, we needed to make sure that the material properties in our database actually followed organized, learnable patterns rather than random noise. 

#### 🧭 Layman's Analogy: The Material Map
Imagine trying to draw a geographical map of materials. Each material has **8 different properties** (like stability, magnetism, atomic sizes, etc.). Because humans cannot visualize 8 dimensions at once, we used a **Variational Autoencoder (VAE)** to compress those 8 properties into a compact 3D coordinate system (similar to coordinates on a globe). We then used **t-SNE** to flatten that globe onto a 2D sheet of paper, making sure similar materials stayed close to each other.

If the database is purely random, coloring the map by a property will look like a scattered mix of confetti. If there are strong physical rules, we will see smooth waves of color (gradients) flowing across the map.

#### 📊 Our Generated Maps (3D Latent Compression):

| Property Map | Physical Interpretation (What it means in simple terms) |
| :--- | :--- |
| ![Bandgap](Results/3_2d_bandgap.png) <br> **Bandgap (eV)** | **Clear Flow**: Shows smooth transitions from purple (low bandgap/conductors) to red (high bandgap/insulators). This tells us that electronic properties are highly organized across the structures. |
| ![Energy Above Hull](Results/3_2d_energy_above_hull.png) <br> **Energy Above Hull (eV)** | **Zoned Clusters**: Stability clusters strongly, showing distinct regions where materials are close to the ground state (highly stable) vs. high-energy zones. |
| ![Formation Energy](Results/3_2d_formation_energy.png) <br> **Formation Energy (eV/atom)** | **Smooth Gradient**: A beautiful, continuous sweep of color indicating that the energy released during material formation is highly predictable from structure coordinates. |
| ![Total Mag](Results/3_2d_total_mag.png) <br> **Total Magnetization ($\mu_B$)** | **Polarized Zones**: Highlights distinct pockets of high magnetization (red) separating from non-magnetic regions (purple/blue), indicating magnetic alignment is physically structured. |
| ![Metallic Radius 2](Results/3_2d_mr2.png) <br> **Metallic Radius B-Site (Å)** | **Structural Patterns**: Demonstrates how the physical size of atoms dictates their coordinates on our map. Larger atoms reside on the left, smaller on the right. |
| ![Metallic Radius 4](Results/3_2d_mr4.png) <br> **Metallic Radius B'-Site (Å)** | **Structural Patterns**: Similar to the B-site, showing clear division by atomic radius, verifying that the spatial size is a primary driver of structure. |
| ![Class Classification](Results/3_2d_class.png) <br> **Class (True vs. Non-True)** | **Confetti (No Pattern)**: The positive (true) and negative (non-true) classifications are mixed randomly. This told us that 'Class' is a noisy target that doesn't follow a clean gradient, which is why we focused our subsequent regression models on predicting the continuous, smoothly-varying physical properties instead. |

### Step 3: Finding Predictability Limits (Linear vs. Decision Trees)
Next, we wanted to know the upper limit of how well we could predict these properties.
- **Linear Models (OLS, Ridge, Lasso)**: Established a simple, transparent baseline. Lasso helped us verify which parameters had non-zero linear contributions.
- **Gradient Boosting (GB) Trees**: Played the role of our "black box" benchmarking. Because decision trees handle complex non-linear combinations and threshold limits naturally, the GB trees showed us the maximum accuracy ($R^2$) we could hope to achieve. Crucially, it gave us a ranking of **Feature Importances**—identifying the exact physical descriptors (e.g., average electronegativity, tolerance factor) that were driving the predictions.

### Step 4: Discovering Physical Laws (Symbolic Regression)
Linear models are easy to read but often miss physics, while Gradient Boosting trees are accurate but impossible to interpret. To combine accuracy with interpretability, we used **Symbolic Regression**.
- We restricted the algorithm to use only basic math operators ($+, -, \times, \div, \sqrt{x}, |x|$).
- To prevent it from generating overly complex equations, we only allowed it to use the top physical descriptors identified by the Gradient Boosting trees in Step 3.
- The algorithm evolved algebraic formulas over thousands of generations until it discovered readable physical equations—such as the custom geometric-electronic Tolerance Factor discovered for double perovskite stability.
