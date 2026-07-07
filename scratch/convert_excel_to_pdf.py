import os
import subprocess
import sys
import pandas as pd

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Double Perovskite Curated Dataset</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        @page {{
            size: A4 landscape;
            margin: 0.4in;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            color: #2D3748;
            font-size: 8pt;
            line-height: 1.3;
        }}
        h1 {{
            font-size: 16pt;
            color: #1A202C;
            margin-top: 0;
            margin-bottom: 5px;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 5px;
        }}
        h2 {{
            font-size: 11pt;
            color: #4A5568;
            margin-top: 15px;
            margin-bottom: 8px;
        }}
        .meta {{
            font-size: 8.5pt;
            margin-bottom: 15px;
            color: #718096;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            page-break-inside: auto;
        }}
        tr {{
            page-break-inside: avoid;
            page-break-after: auto;
        }}
        th, td {{
            border: 1px solid #CBD5E0;
            padding: 5px 7px;
            text-align: left;
        }}
        th {{
            background-color: #EDF2F7;
            color: #2D3748;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #F7FAFC;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
    <h1>Double Perovskite Curated Dataset</h1>
    <div class="meta">Supplementary Material: Curated database of double perovskite compounds and DFT-calculated properties used for training models.</div>
    
    <h2>Sheet 1: True Double Perovskites (N = {len_true} samples)</h2>
    {table_true}
    
    <div class="page-break"></div>
    
    <h1>Double Perovskite Curated Dataset</h1>
    <div class="meta">Supplementary Material: Curated database of double perovskite compounds and DFT-calculated properties used for training models.</div>
    
    <h2>Sheet 2: Non-True Double Perovskites (N = {len_nontrue} samples)</h2>
    {table_nontrue}
</body>
</html>
"""

def make_pdf():
    excel_path = r"data/raw/true and non true double perovskite sort.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}", file=sys.stderr)
        sys.exit(1)
        
    print("Reading sheets from Excel file...")
    df_true = pd.read_excel(excel_path, sheet_name='approx true double perovskite')
    df_nontrue = pd.read_excel(excel_path, sheet_name='nontrue double perovskite')
    
    # Filter columns to only keep important features for printable layout
    cols_to_keep = [
        'Material ID', 'Formula', 'Formation Energy (eV/atom)', 
        'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (μB)', 
        'Space Group', 'element1', 'element3', 'element4'
    ]
    
    # Ensure columns exist before filtering
    cols_true = [c for c in cols_to_keep if c in df_true.columns]
    cols_nontrue = [c for c in cols_to_keep if c in df_nontrue.columns]
    
    df_true_clean = df_true[cols_true].dropna(subset=['Formula']).copy()
    df_nontrue_clean = df_nontrue[cols_nontrue].dropna(subset=['Formula']).copy()
    
    # Format numerical values for presentation
    float_cols = ['Formation Energy (eV/atom)', 'Energy Above Hull (eV)', 'Band Gap (eV)', 'Total Magnetization (μB)']
    for col in float_cols:
        if col in df_true_clean.columns:
            df_true_clean[col] = df_true_clean[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not pd.isna(x) else x)
        if col in df_nontrue_clean.columns:
            df_nontrue_clean[col] = df_nontrue_clean[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not pd.isna(x) else x)
            
    table_true_html = df_true_clean.to_html(index=False, classes='table')
    table_nontrue_html = df_nontrue_clean.to_html(index=False, classes='table')
    
    html_content = HTML_TEMPLATE.format(
        len_true=len(df_true_clean),
        table_true=table_true_html,
        len_nontrue=len(df_nontrue_clean),
        table_nontrue=table_nontrue_html
    )
    
    temp_html = "temp_dataset.html"
    pdf_out = "double_perovskite_dataset.pdf"
    
    print(f"Writing temporary HTML to {temp_html}...")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        print(f"Error: Chrome executable not found at {chrome_path}", file=sys.stderr)
        sys.exit(1)
        
    html_abs_path = os.path.abspath(temp_html)
    pdf_abs_path = os.path.abspath(pdf_out)
    
    print(f"Converting HTML to PDF via Chrome...")
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf_abs_path}",
        f"file:///{html_abs_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully generated dataset PDF: {pdf_out}!")
        success = True
    else:
        print(f"Failed to generate PDF: {result.stderr}", file=sys.stderr)
        success = False
        
    if os.path.exists(temp_html):
        print("Cleaning up temporary HTML file...")
        os.remove(temp_html)
        
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    make_pdf()
