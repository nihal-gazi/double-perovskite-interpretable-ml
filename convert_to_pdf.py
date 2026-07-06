import os
import subprocess
import markdown
import sys

# HTML Template with Google Fonts, custom CSS, KaTeX CDN for math, and Mermaid CDN for diagrams
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <!-- Google Fonts for premium typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- KaTeX CSS for math styling -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #2D3748;
            line-height: 1.6;
            font-size: 11pt;
            margin: 1.5in 1in; /* Page margins for printing */
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #1A202C;
            font-weight: 700;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid; /* Prevent headers separating from content */
        }}
        
        h1 {{
            font-size: 24pt;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 0.3em;
            margin-top: 0;
        }}
        
        h2 {{
            font-size: 16pt;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.2em;
        }}
        
        h3 {{
            font-size: 13pt;
        }}
        
        p {{
            margin-top: 0;
            margin-bottom: 1em;
            text-align: justify;
        }}
        
        /* Table formatting */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1.5em;
            margin-bottom: 1.5em;
            page-break-inside: avoid; /* Avoid breaking tables across pages if possible */
            font-size: 10pt;
        }}
        
        th, td {{
            border: 1px solid #CBD5E0;
            padding: 8px 12px;
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
        
        /* Code blocks and inline code */
        pre {{
            background-color: #F7FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9.5pt;
            margin-top: 1em;
            margin-bottom: 1em;
            page-break-inside: avoid;
        }}
        
        code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10pt;
            background-color: #EDF2F7;
            padding: 2px 4px;
            border-radius: 4px;
            color: #E53E3E;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            color: inherit;
        }}
        
        /* Blockquotes */
        blockquote {{
            border-left: 4px solid #3182CE;
            background-color: #EBF8FF;
            margin-left: 0;
            margin-right: 0;
            padding: 10px 20px;
            font-style: italic;
            border-radius: 0 4px 4px 0;
        }}
        
        /* Lists */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 1em;
            padding-left: 20px;
        }}
        
        li {{
            margin-bottom: 0.4em;
        }}
        
        /* Image formatting */
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5em auto;
            border-radius: 6px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            page-break-inside: avoid;
        }}
        
        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}
    </style>
    
    <!-- KaTeX JS and Auto-Render extension -->
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" 
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}}
                ]
            }});"></script>
</head>
<body>
    {content}
</body>
</html>
"""

def convert_md_to_html(md_filename, html_filename, title):
    print(f"Reading {md_filename}...")
    with open(md_filename, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    # Enable markdown extensions for tables and code blocks
    html_content = markdown.markdown(
        md_text, 
        extensions=["tables", "fenced_code"]
    )
    
    full_html = HTML_TEMPLATE.format(title=title, content=html_content)
    
    print(f"Writing temporary HTML to {html_filename}...")
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(full_html)

def generate_pdf_via_chrome(html_filename, pdf_filename):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        print(f"Error: Chrome executable not found at {chrome_path}", file=sys.stderr)
        return False
        
    html_abs_path = os.path.abspath(html_filename)
    pdf_abs_path = os.path.abspath(pdf_filename)
    
    print(f"Calling Google Chrome to print to PDF: {pdf_filename}...")
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--virtual-time-budget=5000",
        f"--print-to-pdf={pdf_abs_path}",
        f"file:///{html_abs_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully generated {pdf_filename}!")
        return True
    else:
        print(f"Chrome PDF printing failed for {html_filename}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False

def main():
    files_to_convert = [
        ("Final_Results.md", "temp_final_result.html", "final_result.pdf", "Final Results & Analysis"),
        ("methodology.md", "temp_methodology.html", "methodology.pdf", "Project Methodology")
    ]
    
    for md, temp_html, pdf, title in files_to_convert:
        if not os.path.exists(md):
            print(f"Warning: {md} not found, skipping.", file=sys.stderr)
            continue
            
        convert_md_to_html(md, temp_html, title)
        success = generate_pdf_via_chrome(temp_html, pdf)
        
        # Clean up temporary HTML
        if os.path.exists(temp_html):
            print(f"Cleaning up {temp_html}...")
            os.remove(temp_html)
            
        if not success:
            sys.exit(1)
            
    print("\n[SUCCESS] Both PDFs have been generated successfully!")

if __name__ == "__main__":
    main()
