<h1 align="center">Interpretable Machine Learning and Symbolic Regression for Predicting Electronic, Magnetic, and Thermodynamic Properties of Double Perovskites</h1>

<p align="center">
  <b>Nihal Gazi</b><sup>1</sup>, 
  <b>Meghneel Dutta</b><sup>2</sup>, 
  <b>Subarna Dutta</b><sup>3,*</sup>, 
  <b>Soumyadipta Pal</b><sup>3,*</sup>
</p>

<p align="center">
  <sup>1</sup><i>Department of Computer Science Engineering and Artificial Intelligence & Machine Learning, Institute of Engineering and Management, Kolkata - 700091, West Bengal, India</i><br>
  <sup>2</sup><i>Department of Electronics and Communication Engineering, Institute of Engineering and Management, Kolkata - 700091, West Bengal, India</i><br>
  <sup>3</sup><i>Institute of Engineering and Management, D-1, Sector-V, Saltlake Electronics Complex, Kolkata - 700091, West Bengal, India</i><br>
  <sup>*</sup>Corresponding Authors: <a href="mailto:subarna.iem87@gmail.com">subarna.iem87@gmail.com</a>, <a href="mailto:soumyadipta.pal@gmail.com">soumyadipta.pal@gmail.com</a>
</p>

<p align="center">
  <a href="paper_v2/paper.pdf"><b>[Read Paper PDF]</b></a> | 
  <a href="run_pipeline.py"><b>[Run Code]</b></a> |
  <a href="data/processed"><b>[Dataset Cache]</b></a>
</p>

---

## 📖 Abstract

Predicting the structural, thermodynamic, electronic, and magnetic properties of complex double perovskite oxides ($A_2BB'O_6$) is critical for designing next-generation functional materials. While deep Graph Neural Networks (GNNs) achieve high accuracy, they operate as "black boxes" lacking physical interpretability. In this work, we present a hybrid materials informatics framework combining linear models (OLS, Ridge, Lasso), Gradient Boosting Regressors (GBR), and Genetic Programming (GP) Symbolic Regression to predict target properties: Formation Energy ($E_f$), Total Magnetization ($M$), Band Gap ($E_g$), and Energy Above Hull ($E_{\text{hull}}$). 

Our pipeline succeeds in distilling deep representation metrics down to simple, factorable algebraic equations that link crystallographic strain (Goldschmidt tolerance factor) and atomic descriptors (electronegativities, valence electron counts) to structural stability.

---

## 📂 Project Directory Structure

```text
Prof SOP sir/
├── data/                                 # Organized dataset folder
│   ├── raw/                              # Original input spreadsheets (Excel format)
│   ├── processed/                        # Query-cached CSV tables (CHGNet, 3D structure, Mendeleev)
│   └── CIF_Files/                        # Crystallographic structure files (.cif)
│
├── src/                                  # Modular codebase source
│   ├── __init__.py
│   ├── config.py                         # Constants, dictionaries (valence, oxidation) and paths
│   ├── data_processing.py                # Sourcing features, attaching d-electrons proxy, etc.
│   ├── VAE.py                            # VAE PyTorch model training and latent space extraction
│   ├── baseline_models.py                # Fits baseline models (OLS, Ridge, Lasso, and GB Trees)
│   ├── symbolic_regression.py            # Fits Symbolic Regression equations matching Table 4
│   └── plots.py                          # Plots correlation heatmaps, diagnostic scatter plots, etc.
│
├── paper_v2/                             # Self-contained LaTeX project directory
│   ├── paper.tex                         # LaTeX source code for the paper draft
│   ├── paper.pdf                         # Compiled PDF version of the paper draft
│   └── *.png, *.bst, *.sty               # Associated figures and style assets
│
├── legacy/                               # Moved legacy exploratory code (for history preservation)
│   ├── Analysis_v1/
│   ├── Analysis_v2/
│   ├── Primary_Analysis/
│   └── Visualize/
│
├── requirements.txt                      # Dependencies required to execute
├── run_pipeline.py                       # Single command pipeline to reproduce all figures and metrics
├── convert_to_pdf.py                     # Headless Chrome PDF report printing utility
└── README.md                             # This file (overview, layout, and instructions)
```

---

## 💻 Prerequisites & Setup

### 1. System Requirements
- **Python**: Python 3.8 to 3.11 is recommended.
- **LaTeX Compiler**: MiKTeX or TeX Live is required to re-compile the paper PDF.

### 2. Installation
We recommend setting up a virtual environment (e.g., venv or conda) before installing dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Unix/macOS

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> PyTorch is required for VAE training. If you have a CUDA-compatible GPU, ensure you install the CUDA version of PyTorch for faster model training. Otherwise, the script will automatically fallback to CPU.

---

## ⚙️ How to Run the Pipeline

### 1. Fast Verification Mode (Recommended)
To verify reproducibility in under 5 seconds, run the script with the `--fast` flag. This mode loads the pre-trained t-SNE coordinates instead of training the PyTorch VAE model for 1000 epochs:
```bash
python run_pipeline.py --fast
```

### 2. Full Run Mode
To run the full end-to-end pipeline (including training the PyTorch VAE model for 1000 epochs, running t-SNE coordinate projections, executing baseline regressions, and fitting genetic programming models):
```bash
python run_pipeline.py
```

### 3. Pipeline Output
When running, the script executes the following stages:
- **Stage 1**: Loads raw excel spreadsheets and processed CSV tables, merges them, and computes structural/quantum features.
- **Stage 2**: Trains/loads the VAE and runs t-SNE, saving scatter plots of the latent space colorcoded by properties.
- **Stage 3**: Fits OLS, Ridge, Lasso, and Gradient Boosting Regressors, saving a 2x2 regression diagnostics plot for Formation Energy.
- **Stage 4**: Runs symbolic regression genetic programming to evolve math equations.
- **Console Printout**: Prints LaTeX-style formatting for **Table 2** (model comparison), **Table 3** (method comparison), and **Table 4** (discovered equations).
- **Output Files**: Generated plots are saved directly to the `Results/` folder.

---

## 📄 How to Compile the LaTeX Paper

To re-compile the paper draft and compile references/figure files into a PDF, follow these steps in your terminal:

```bash
# Navigate to the LaTeX project directory
cd paper_v2

# Run pdflatex first to generate aux files
pdflatex paper.tex

# Compile the bibliography
bibtex paper

# Run pdflatex twice more to resolve citations and figure numbers
pdflatex paper.tex
pdflatex paper.tex
```

---

## 📊 Summary of Baseline Model Performance (R² & MAE)
Running the pipeline reproduces the exact benchmarks in the paper:

| Target Property | OLS R² (MAE) | Ridge R² (MAE) | Lasso R² (MAE) | GBR R² (MAE) |
| :--- | :---: | :---: | :---: | :---: |
| **Formation Energy (eV/atom)** | 0.6169 (0.1766) | 0.6227 (0.1772) | **0.6263** (0.1778) | 0.6841 (0.1559)* |
| **Total Magnetization ($\mu\_B$)** | **0.6019** (2.7835) | 0.5960 (2.7881) | 0.6009 (2.7699) | 0.6061 (2.8001) |
| **Band Gap (eV)** | -0.2372 (0.7678) | -0.2342 (0.7672) | **-0.2090** (0.7685) | 0.1808 (0.6438) |
| **Energy Above Hull (eV)** | **0.3610** (0.0464) | 0.3491 (0.0478) | 0.3290 (0.0483) | 0.3424 (0.0400) |

*\*Note: The GBR Formation Energy score of 0.6841 is an improved model compared to the paper's reported 0.6247, achieved using the expanded 27-descriptor dataset.*

---

## 📐 Best Discovered Physical Equations (Table 4)

1. **Formation Energy ($E\_f$):**

$$
\Delta E\_f = 0.1508 \cdot \text{Val}\_{\text{avg}} - 1.0197 \text{EN}\_A + 0.4113 \text{EN}\_{\text{avg}} - 0.2725 \text{Shannon}\_A - 2.6972
$$

2. **Total Magnetization ($M$):**

$$
M = \left| \frac{t}{E\_{\text{GNN}}} + E\_{\text{GNN}} + \text{EN}\_{\text{avg}} + \left| \frac{\text{EN}\_{\text{avg}}}{E\_{\text{GNN}}} + E\_{\text{GNN}} + 2 \text{EN}\_{\text{avg}} + M\_{\text{abs}} - 0.249 \right| \right|
$$

3. **Energy Above Hull ($E\_{\text{hull}}$):**

$$
E\_{\text{hull}} = -0.142 \cdot E\_{\text{GNN}}
$$

4. **Band Gap ($E\_g$):**

$$
E\_g = E\_{\text{GNN}} + 0.980
$$
