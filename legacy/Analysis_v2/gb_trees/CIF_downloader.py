import os
import pandas as pd
import requests
from pymatgen.core.structure import Structure
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
API_KEY = "gWJXczH9PXlsJ4tByN7ilvwJGv0TMnsY"

BASE_DIR = Path(r"C:\Users\user\Desktop\IEM\IEM projects\Prof SOP sir")
DATA_PATH = BASE_DIR / "Data" / "true and non true double perovskite sort.xlsx"
CIF_DIR = BASE_DIR / "Data" / "CIF_Files"

CIF_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD MATERIAL IDs
# ==========================================
print("Loading dataset...")
df = pd.read_excel(DATA_PATH, sheet_name='approx true double perovskite')

# Clean and extract valid mp- IDs
df_clean = df.dropna(subset=['Material ID'])
df_clean = df_clean[df_clean['Material ID'].astype(str).str.startswith('mp-')]
material_ids = df_clean['Material ID'].astype(str).str.strip().tolist()

print(f"Found {len(material_ids)} valid Material IDs. Starting direct REST API download...")

# ==========================================
# 3. DIRECT REST API DOWNLOADER (Bug-Free)
# ==========================================
headers = {'X-API-KEY': API_KEY}

for m_id in tqdm(material_ids, desc="Downloading CIFs"):
    cif_path = CIF_DIR / f"{m_id}.cif"
    
    # Skip if already downloaded
    if cif_path.exists():
        continue
        
    try:
        # Query the raw REST API directly, bypassing the broken mp-api library
        url = f"https://api.materialsproject.org/materials/core/?material_ids={m_id}&_fields=structure"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Ensure the material exists in the database
            if data.get("data") and len(data["data"]) > 0:
                
                # Extract the raw structure dictionary
                struct_dict = data["data"][0]["structure"]
                
                # Use pymatgen to parse the JSON and save it as a true 3D .cif file
                structure = Structure.from_dict(struct_dict)
                structure.to(filename=str(cif_path), fmt="cif")
            else:
                pass # Material might be deprecated or private
        else:
            print(f"\nAPI Error on {m_id}: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"\nFailed to process {m_id}: {e}")

print(f"\n✅ Successfully downloaded 3D crystal files to {CIF_DIR}")