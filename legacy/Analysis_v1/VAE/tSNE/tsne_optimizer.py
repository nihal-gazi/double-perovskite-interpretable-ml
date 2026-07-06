import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import joblib

# ==========================================
# 1. ABSOLUTE PATHS & SETUP
# ==========================================
BASE_DIR = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir"
DATA_PATH = os.path.join(BASE_DIR, "Data", "true and non true double perovskite sort.xlsx")
VAE_MODEL_DIR = os.path.join(BASE_DIR, "Models", "VAE")
TSNE_MODEL_DIR = os.path.join(BASE_DIR, "Models", "tSNE")

# Ensure t-SNE output directory exists
os.makedirs(TSNE_MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==========================================
# 2. LOAD DATA & VAE ASSETS
# ==========================================
features = [
    'metallic radius1', 'metallic radius 2', 'metallic radius 3', 'metallic radius4',
    'Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (Î¼B)'
]

print("Loading data...")
df_true = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')
df_nontrue = pd.read_excel(DATA_PATH, sheet_name='nontrue double perovskite')

df_true['Class'] = 'True Perovskite'
df_nontrue['Class'] = 'Non-True'

df_combined = pd.concat([df_true, df_nontrue], ignore_index=True)

for col in features:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

df_clean = df_combined.dropna(subset=['Formation Energy (eV/atom)']).copy()
df_clean[features] = df_clean[features].fillna(0)

# Load the scaler used during VAE training
scaler = joblib.load(os.path.join(VAE_MODEL_DIR, 'scaler.pkl'))
X_scaled = scaler.transform(df_clean[features].values)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# ==========================================
# 3. VAE ENCODING (Extracting Latent Space)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h) # We only need the mean (mu) for extraction

# Find latest VAE model
model_files = glob.glob(os.path.join(VAE_MODEL_DIR, "*.pth"))
latest_vae = max(model_files, key=os.path.getctime)

print(f"Loading VAE weights from: {os.path.basename(latest_vae)}")
vae = VAE(input_dim=len(features), latent_dim=64).to(device)
vae.load_state_dict(torch.load(latest_vae, map_location=device), strict=False)
vae.eval()

with torch.no_grad():
    latent_5d = vae.encode(X_tensor).cpu().numpy()

# ==========================================
# 4. t-SNE OPTIMIZATION (N=3)
# ==========================================
print("Running t-SNE to reduce VAE manifold from 5D to 3D...")
# Perplexity usually kept between 5 and 50 based on dataset size
tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
latent_3d = tsne.fit_transform(latent_5d)

# Save results to DataFrame
df_tsne = pd.DataFrame(latent_3d, columns=['tSNE_1', 'tSNE_2', 'tSNE_3'])
df_tsne['Class'] = df_clean['Class'].values
df_tsne['Formula'] = df_clean['Formula'].values

# Attach original properties so we can color-code the visualizer later
for col in features:
    df_tsne[col] = df_clean[col].values

# Save the embedding
output_file = os.path.join(TSNE_MODEL_DIR, "tsne_embeddings.pkl")
df_tsne.to_pickle(output_file)
print(f"✅ t-SNE embeddings successfully saved to: {output_file}")