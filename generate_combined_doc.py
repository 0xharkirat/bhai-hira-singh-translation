import os
import re

def parse_md_file(filepath):
    """Extracts content from a markdown file, ignoring frontmatter."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove frontmatter (between --- and ---)
            content = re.sub(r'^---\n.*?---\n', '', content, flags=re.DOTALL)
            return content.strip()
    except Exception as e:
        return f"Error reading file: {e}"

def generate_html_doc(pa_dir, en_dir, output_file):
    """Generates a two-column HTML file from paired MD files."""
    
    # Get all page numbers
    pa_files = sorted([f for f in os.listdir(pa_dir) if f.endswith('.pa.md')])
    pages = []
    
    for f in pa_files:
        # Extract page number (assuming format page_XXXX.pa.md)
        match = re.search(r'page_(\d+)', f)
        if match:
            pages.append(match.group(1))
            
    html_content = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Bhai Hira Singh Ji - Combined Translation</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            table { width: 100%; border-collapse: collapse; table-layout: fixed; }
            th, td { border: 1px solid #ddd; padding: 15px; vertical-align: top; word-wrap: break-word; }
            th { background-color: #f2f2f2; }
            .pa-col { width: 50%; font-family: 'Raavi', 'Arial Unicode MS', sans-serif; }
            .en-col { width: 50%; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .page-header { background-color: #e0e0e0; font-weight: bold; text-align: center; }
        </style>
    </head>
    <body>
        <h1>Bhai Hira Singh Ji - Combined Translation</h1>
        <table>
            <thead>
                <tr>
                    <th>Punjabi (Original/OCR)</th>
                    <th>English (Translation)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for page_num in pages:
        pa_path = os.path.join(pa_dir, f"page_{page_num}.pa.md")
        en_path = os.path.join(en_dir, f"page_{page_num}.en.md")
        
        pa_text = parse_md_file(pa_path)
        en_text = parse_md_file(en_path) if os.path.exists(en_path) else "*(Translation not available)*"
        
        # Convert markdown newlines to HTML breaks for basic formatting
        pa_html = pa_text.replace('\n', '<br>')
        en_html = en_text.replace('\n', '<br>')
        
        # Simple markdown headers to bold
        pa_html = re.sub(r'#{1,6}\s+(.*?)<br>', r'<strong>\1</strong><br>', pa_html)
        en_html = re.sub(r'#{1,6}\s+(.*?)<br>', r'<strong>\1</strong><br>', en_html)
        
        html_content += f"""
                <tr>
                    <td colspan="2" class="page-header">Page {page_num}</td>
                </tr>
                <tr>
                    <td class="pa-col">{pa_html}</td>
                    <td class="en-col">{en_html}</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    PA_DIR = "out/pa"
    EN_DIR = "out/en"
    OUTPUT_FILE = "combined_translation.html"
    generate_html_doc(PA_DIR, EN_DIR, OUTPUT_FILE)
