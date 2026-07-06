import os
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
LATENT_DIM = 3 # Increased to 5 so we have axes (3) + sliders (2)
DATA_PATH = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Data\true and non true double perovskite sort.xlsx"
MODEL_DIR = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Models\VAE"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
df_true = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')
df_nontrue = pd.read_excel(DATA_PATH, sheet_name='nontrue double perovskite')
df_combined = pd.concat([df_true, df_nontrue], ignore_index=True)

features = [
    'metallic radius1', 'metallic radius 2', 'metallic radius 3', 'metallic radius4',
    'Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (Î¼B)'
]

for col in features:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

df_clean = df_combined.dropna(subset=['Formation Energy (eV/atom)'])
df_clean[features] = df_clean[features].fillna(0)

X_numpy = df_clean[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numpy)
# Save scaler for the visualizer
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
dataloader = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=True)

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
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

model = VAE(input_dim=len(features), latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)

def vae_loss(reconstructed, original, mu, logvar):
    mse = nn.MSELoss(reduction='sum')(reconstructed, original)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

# ==========================================
# 4. TRAINING & SAVING
# ==========================================
epochs = 1000
model.train()
set = "["
for epoch in range(epochs):
    for batch in dataloader:
        batch_x = batch[0]
        optimizer.zero_grad()
        reconstructed, mu, logvar = model(batch_x)
        loss = vae_loss(reconstructed, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} Complete. Loss = {loss}")
        set += str(loss.item())+","

print(set+"]")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(MODEL_DIR, f"model-{timestamp}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model successfully saved to: {model_path}")