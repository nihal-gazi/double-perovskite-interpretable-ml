import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="VAE -> t-SNE Explorer")
st.title("🌌 VAE + t-SNE 3D Projection")
st.markdown("Visualizing the 5D VAE Latent Space reduced to 3D via t-SNE.")

TSNE_MODEL_DIR = r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir\Models\tSNE"
DATA_FILE = os.path.join(TSNE_MODEL_DIR, "tsne_embeddings.pkl")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Cannot find t-SNE data at {DATA_FILE}. Please run tSNE_optimizer.py first.")
        st.stop()
    return pd.read_pickle(DATA_FILE)

df_tsne = load_data()

# ==========================================
# 2. UI CONTROLS
# ==========================================
st.sidebar.header("Visualization Controls")

# Let the user choose what to color the 3D dots by
color_options = ['Class'] + [col for col in df_tsne.columns if col not in ['tSNE_1', 'tSNE_2', 'tSNE_3', 'Formula', 'Class']]
color_feature = st.sidebar.selectbox("Color the points by:", color_options)

st.sidebar.markdown("""
**How this works:**
1. Raw Data $\\rightarrow$ 
2. VAE Encoder (5D Latent) $\\rightarrow$ 
3. t-SNE (3D Projection)
""")

# ==========================================
# 3. PLOTTING
# ==========================================
# Determine coloring logic (discrete for Class, continuous for physical properties)
if color_feature == 'Class':
    fig = px.scatter_3d(
        df_tsne, x='tSNE_1', y='tSNE_2', z='tSNE_3',
        color='Class',
        hover_name='Formula',
        color_discrete_sequence=['#00CC96', '#EF553B'],
        title="t-SNE Projection: True vs Non-True Perovskites"
    )
else:
    fig = px.scatter_3d(
        df_tsne, x='tSNE_1', y='tSNE_2', z='tSNE_3',
        color=color_feature,
        hover_name='Formula',
        color_continuous_scale='Turbo',
        title=f"t-SNE Projection Colored by {color_feature}"
    )

# Tweak visuals for a cleaner look
fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=40))

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)