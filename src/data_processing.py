import pandas as pd
import numpy as np
from src.config import CHGNET_CACHE, THREE_D_CACHE, ORIGINAL_DATA, A_OXIDATION_DICT, TM_VALENCE_DICT

def get_a_ox(el):
    return A_OXIDATION_DICT.get(str(el).strip(), np.nan)

def get_tm_val(el):
    return TM_VALENCE_DICT.get(str(el).strip(), np.nan)

def load_and_preprocess_data():
    if not CHGNET_CACHE.exists():
        raise FileNotFoundError(f"Missing CHGNet cache at: {CHGNET_CACHE}")
    if not THREE_D_CACHE.exists():
        raise FileNotFoundError(f"Missing 3D cache at: {THREE_D_CACHE}")
        
    df_chgnet = pd.read_csv(CHGNET_CACHE)
    df_3d = pd.read_csv(THREE_D_CACHE)
    
    # Rename columns to standard formats
    rename_dict = {}
    for col in df_chgnet.columns:
        if 'Total Magnetization' in col:
            rename_dict[col] = 'Total Magnetization (uB)'
        elif 'Band Gap' in col:
            rename_dict[col] = 'Band Gap (eV)'
        elif 'Energy Above Hull' in col:
            rename_dict[col] = 'Energy Above Hull (eV)'
        elif 'Formation Energy' in col:
            rename_dict[col] = 'Formation Energy (eV/atom)'
            
    df_chgnet = df_chgnet.rename(columns=rename_dict)
    df_3d = df_3d.rename(columns=rename_dict)
    
    # Merge datasets
    df_merged = pd.merge(df_chgnet, df_3d[['Material ID', '3D_Volume_Per_Atom', '3D_Density', '3D_Crystal_Symmetry']], on='Material ID', how='inner')
    
    # Load raw excel to get elements for quantum spin calculations
    df_orig = pd.read_excel(ORIGINAL_DATA, sheet_name='approx true double perovskite')
    
    # Clean Material ID mapping
    df_orig['Material ID'] = df_orig['Material ID'].astype(str).str.strip()
    df_merged['Material ID'] = df_merged['Material ID'].astype(str).str.strip()
    
    # Map elements and metallic radii back
    elem_cols = [
        'element1', 'metallic radius1', 'element2', 'metallic radius 2',
        'element3', 'metallic radius 3', 'element4', 'metallic radius4'
    ]
    elem_map = df_orig.set_index('Material ID')[elem_cols]
    # Drop columns that are already present in df_merged before joining
    cols_to_join = [c for c in elem_cols if c not in df_merged.columns]
    df_merged = df_merged.join(elem_map[cols_to_join], on='Material ID', how='left')
    
    # Calculate physical / quantum spin descriptors
    df_merged['Total_A_Charge'] = 2 * df_merged['element1'].apply(get_a_ox)
    df_merged['Total_d_electrons'] = (df_merged['element3'].apply(get_tm_val) + df_merged['element4'].apply(get_tm_val)) - (12 - df_merged['Total_A_Charge'])
    df_merged['Spin_Proxy_Distance'] = np.abs(df_merged['Total_d_electrons'] - 5)
    
    O_SHANNON = 1.40
    df_merged['d_AO'] = pd.to_numeric(df_merged['metallic radius1'], errors='coerce') + O_SHANNON
    df_merged['d_BO'] = df_merged['Shannon_B'] + O_SHANNON
    df_merged['d_BprimeO'] = df_merged['Shannon_Bprime'] + O_SHANNON
    df_merged['d_avg'] = (df_merged['d_BO'] + df_merged['d_BprimeO']) / 2
    
    # Standardize space group to numeric
    df_merged['3D_Crystal_Symmetry'] = pd.to_numeric(df_merged['3D_Crystal_Symmetry'], errors='coerce').fillna(0)
    
    return df_merged
