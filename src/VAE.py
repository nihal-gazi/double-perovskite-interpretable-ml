import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import joblib
from src.config import ORIGINAL_DATA, MODELS_DIR

# Features used for VAE compression
VAE_FEATURES = [
    'metallic radius1', 'metallic radius 2', 'metallic radius 3', 'metallic radius4',
    'Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (uB)'
]

class VAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(VAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h)

def vae_loss(reconstructed, original, mu, logvar):
    mse = nn.MSELoss(reduction='sum')(reconstructed, original)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

def train_and_project_vae(epochs=1000):
    print("Loading data for VAE...")
    df_true = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')
    df_nontrue = pd.read_excel(ORIGINAL_DATA, sheet_name='nontrue double perovskite')
    
    df_true['Class'] = 'True Perovskite'
    df_nontrue['Class'] = 'Non-True'
    
    df_combined = pd.concat([df_true, df_nontrue], ignore_index=True)
    
    # Standardize column names
    rename_dict = {}
    for col in df_combined.columns:
        if 'Total Magnetization' in col:
            rename_dict[col] = 'Total Magnetization (uB)'
        elif 'Band Gap' in col:
            rename_dict[col] = 'Band Gap (eV)'
        elif 'Energy Above Hull' in col:
            rename_dict[col] = 'Energy Above Hull (eV)'
        elif 'Formation Energy' in col:
            rename_dict[col] = 'Formation Energy (eV/atom)'
    df_combined = df_combined.rename(columns=rename_dict)
    
    for col in VAE_FEATURES:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
    df_clean = df_combined.dropna(subset=['Formation Energy (eV/atom)']).copy()
    df_clean[VAE_FEATURES] = df_clean[VAE_FEATURES].fillna(0)
    
    X_numpy = df_clean[VAE_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numpy)
    
    # Save VAE artifacts
    vae_model_dir = MODELS_DIR / "VAE"
    vae_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, vae_model_dir / 'scaler.pkl')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on device: {device}")
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=True)
    
    model = VAEModel(input_dim=len(VAE_FEATURES), latent_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch_x = batch[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_x)
            loss = vae_loss(reconstructed, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.2f}")
            
    # Save trained VAE model parameters
    model_path = vae_model_dir / "vae_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"VAE model saved to {model_path}")
    
    # Latent space extraction
    model.eval()
    with torch.no_grad():
        latent_3d = model.encode(X_tensor).cpu().numpy()
        
    # Project 3D Latent Space to 2D using t-SNE
    print("Running t-SNE (reducing 3D latent space to 2D)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    latent_2d = tsne.fit_transform(latent_3d)
    
    df_tsne = pd.DataFrame(latent_2d, columns=['tSNE_1', 'tSNE_2'])
    df_tsne['Class'] = df_clean['Class'].values
    df_tsne['Formula'] = df_clean['Formula'].values
    for col in VAE_FEATURES:
        df_tsne[col] = df_clean[col].values
        
    tsne_model_dir = MODELS_DIR / "tSNE"
    tsne_model_dir.mkdir(parents=True, exist_ok=True)
    output_file = tsne_model_dir / "tsne_embeddings_2d.pkl"
    df_tsne.to_pickle(output_file)
    print(f"t-SNE 2D embeddings successfully saved to: {output_file}")
    
    return df_tsne
