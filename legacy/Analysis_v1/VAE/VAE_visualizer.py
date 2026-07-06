import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import glob

# ==========================================
# 1. SETUP & UTILS
# ==========================================
st.set_page_config(layout="wide", page_title="VAE 2D Heatmap Explorer")
st.title("🌌 VAE 2D Latent Heatmap Explorer")

MODEL_DIR = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Models\VAE"
LATENT_DIM = 5
features = [
    'metallic radius1', 'metallic radius 2', 'metallic radius 3', 'metallic radius4',
    'Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (Î¼B)'
]

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def decode(self, z):
        return self.decoder(z)

@st.cache_resource
def load_assets():
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    if not model_files:
        st.error("No model found. Run VAE_trainer.py first.")
        st.stop()
    latest_model = max(model_files, key=os.path.getctime)
    
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    model = VAE(input_dim=len(features), latent_dim=LATENT_DIM)
    
    state_dict = torch.load(latest_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False) 
    model.eval()
    return model, scaler

model, scaler = load_assets()

# ==========================================
# 2. UI CONTROLS (Axes Mapping & Sliders)
# ==========================================
st.sidebar.header("Latent Space Controls")

axis_mapping = {}
slider_values = {}

# We only want X and Y for a 2D Heatmap
for i in range(LATENT_DIM):
    st.sidebar.markdown(f"**Latent Dimension {i+1}**")
    col1, col2 = st.sidebar.columns([1, 1])
    
    choice = col1.radio("Map to:", ["X", "Y", "Slider"], key=f"radio_{i}", horizontal=True)
    
    if choice in ["X", "Y"]:
        axis_mapping[choice] = i
    else:
        slider_values[i] = col2.slider("Value", -3.0, 3.0, 0.0, step=0.1, key=f"slider_{i}")
    st.sidebar.divider()

if len(axis_mapping) != 2 or len(set(axis_mapping.values())) != 2:
    st.warning("⚠️ Please map exactly ONE unique dimension to X, and ONE to Y.")
    st.stop()

st.sidebar.header("Visualization Target")
color_feature = st.sidebar.selectbox("Show Heatmap for:", features)

# ==========================================
# 3. HIGH-RES MANIFOLD GENERATION
# ==========================================
# Increased resolution to 60x60 for a smooth 2D heatmap
resolution = 60
grid_1d = np.linspace(-3, 3, resolution)
X_grid, Y_grid = np.meshgrid(grid_1d, grid_1d)

X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()

Z_input = np.zeros((len(X_flat), LATENT_DIM))

# Assign the meshgrid to the chosen X and Y latent dimensions
Z_input[:, axis_mapping["X"]] = X_flat
Z_input[:, axis_mapping["Y"]] = Y_flat

# Assign the static slider values to the remaining dimensions
for dim_idx, val in slider_values.items():
    Z_input[:, dim_idx] = val

# Decode the physics
with torch.no_grad():
    Z_tensor = torch.tensor(Z_input, dtype=torch.float32)
    reconstructed_scaled = model.decode(Z_tensor).numpy()

reconstructed_real = scaler.inverse_transform(reconstructed_scaled)
df_reconstructed = pd.DataFrame(reconstructed_real, columns=features)

# Reshape the targeted feature back into a 2D grid for the heatmap
heatmap_z_values = df_reconstructed[color_feature].values.reshape(resolution, resolution)

# ==========================================
# 4. 2D CONTOUR PLOTTING
# ==========================================
fig = go.Figure(data=go.Contour(
    z=heatmap_z_values,
    x=grid_1d, # Latent X axis
    y=grid_1d, # Latent Y axis
    colorscale='Turbo',
    contours=dict(
        showlines=True
    ),

    line=dict(
        color='black',
        width=0.5
    ),
    colorbar=dict(title=color_feature.split('(')[0].strip()) # Clean up colorbar title
))

fig.update_layout(
    title=f"Latent Manifold Heatmap: {color_feature}",
    xaxis_title=f"Latent Dimension {axis_mapping['X'] + 1}",
    yaxis_title=f"Latent Dimension {axis_mapping['Y'] + 1}",
    height=750,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)