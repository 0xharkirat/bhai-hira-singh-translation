import os
import sys
import argparse
import csv
import time
import glob
import re
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, api_key, model_name, temperature):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=1.0,
            )
        )

    def translate_page(self, text, page_info):
        """
        Translates a single page using Gemini.
        """
        page_type = page_info.get('page_type', 'TEXT_PAGE')
        confidence = float(page_info.get('mean_confidence', 0))
        
        prompt = self._construct_prompt(text, page_type, confidence)
        
        # Retry logic for transient errors
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except exceptions.ResourceExhausted as e:
                # Try to extract retry delay from error message, default to exponential backoff
                retry_delay = float(re.search(r'retry in (\d+\.?\d*)s', str(e)).group(1)) if re.search(r'retry in (\d+\.?\d*)s', str(e)) else base_delay * (2 ** attempt)
                # Cap at 60s or use the reported delay if larger
                wait_time = max(retry_delay, base_delay * (2 ** attempt))
                if wait_time < 30 and attempt > 0: # minimal wait on 2nd attempt for 429
                    wait_time = 30
                
                logger.warning(f"Rate limit hit. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay)
                
        return None

    def _construct_prompt(self, text, page_type, confidence):
        instructions = (
            "You are a professional interpreter translating Punjabi OCR text to English.\n"
            "Your goal is meaning-preserving translation that captures the tone and nuance, not a literal word-for-word conversion.\n\n"
            f"CONTEXT:\n"
            f"Page Type: {page_type}\n"
            f"OCR Confidence: {confidence}\n\n"
            "RULES:\n"
            "1. Preserve meaning and tone.\n"
            "2. Fix obvious OCR spelling issues ONLY if the context makes it clear (e.g., missing vowels).\n"
            "3. If text is unclear, use [UNCLEAR]. Do not invent facts.\n"
            "4. Keep consistent transliteration for names/places/people.\n"
            "5. Preserve formatting, headings, and line breaks for poetry/verse.\n"
        )
        
        if page_type == 'FIGURE_PAGE':
            instructions += "6. This is a Figure/Image page. Translate only short captions or labels if present. Ignore random noise.\n"
            
        if confidence < 60:
            instructions += "6. WARNING: Low OCR confidence. Be extra careful, text may be garbled. Mark highly uncertain parts as [UNCLEAR].\n"

        input_section = f"\nINPUT PUNJABI TEXT:\n{text}\n"
        
        output_format = (
            "\nOUTPUT FORMAT:\n"
            "Provide the output in Markdown.\n"
            "Structure your response exactly as follows:\n\n"
            "## English (Meaning Translation)\n"
            "<English translation here>\n\n"
            "## Notes\n"
            "- Corrected likely OCR word: \"<ocr>\" -> \"<intended>\" (reason: context)\n"
            "(Keep notes section empty if no corrections made, or omit it if none needed. Max 5 items.)\n"
        )
        
        return instructions + input_section + output_format

def parse_frontmatter(content):
    """
    Parses YAML frontmatter from markdown content.
    Returns (frontmatter_dict, body_text)
    """
    pattern = r"^---\n(.*?)\n---\n"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        fm_text = match.group(1)
        try:
            fm = yaml.safe_load(fm_text)
            body = content[match.end():]
            return fm, body
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
            return {}, content
    
    return {}, content

def load_report(report_path):
    """
    Loads the OCR report CSV into a dictionary keyed by page_index.
    """
    report = {}
    if not os.path.exists(report_path):
        logger.warning(f"Report file not found: {report_path}")
        return report
        
    with open(report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['page_index'])
                report[idx] = row
            except ValueError:
                continue
    return report

def get_page_index_from_filename(filename):
    """
    Extracts page index from filename page_0001.pa.md
    """
    match = re.search(r'page_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Translate Punjabi OCR files using Gemini.")
    parser.add_argument("--pa", required=True, help="Input directory containing Punjabi markdown files")
    parser.add_argument("--report", required=True, help="Path to OCR report CSV")
    parser.add_argument("--out", required=True, help="Output directory root")
    parser.add_argument("--start", type=int, default=1, help="Start page index")
    parser.add_argument("--end", type=int, default=None, help="End page index")
    parser.add_argument("--force", action='store_true', help="Overwrite existing files")
    parser.add_argument("--dry-run", action='store_true', help="Dry run mode")
    parser.add_argument("--model", default="gemini-flash-latest", help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    
    args = parser.parse_args()
    
    # Load environment variables from .env if present
    load_dotenv()
    
    # Setup directories
    out_en_dir = os.path.join(args.out, "en")
    out_bi_dir = os.path.join(args.out, "bilingual")
    os.makedirs(out_en_dir, exist_ok=True)
    os.makedirs(out_bi_dir, exist_ok=True)
    
    # Check API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    # Load Data
    report_data = load_report(args.report)
    input_files = sorted(glob.glob(os.path.join(args.pa, "*.md")))
    
    translator = Translator(api_key, args.model, args.temperature)
    
    # Prepare output report
    out_report_path = os.path.join(args.out, "translate_report.csv")
    report_fieldnames = ['page_index', 'status', 'translation_model', 'notes']
    
    # Initialize report file if it doesn't exist or we are forcing? 
    # Actually simpler to append or rewrite. Let's just append for now creates duplicate execution issues but safer.
    # Or load existing report to skip? For now we just write a new one or append? 
    # Let's write line by line.
    
    results = []
    
    logger.info(f"Found {len(input_files)} input files.")
    
    for file_path in tqdm(input_files):
        idx = get_page_index_from_filename(os.path.basename(file_path))
        if idx is None:
            continue
            
        if idx < args.start:
            continue
        if args.end and idx > args.end:
            continue
            
        page_info = report_data.get(idx, {})
        
        # Define output paths
        out_en_path = os.path.join(out_en_dir, f"page_{idx:04d}.en.md")
        out_bi_path = os.path.join(out_bi_dir, f"page_{idx:04d}.md")
        
        if os.path.exists(out_en_path) and not args.force:
            results.append({'page_index': idx, 'status': 'skipped', 'translation_model': args.model, 'notes': 'File exists'})
            continue
            
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        fm, body_text = parse_frontmatter(content)
        
        if not body_text.strip():
            logger.info(f"Page {idx} is empty, skipping translation call.")
            # Still copy to output as empty?
            # User wants translations. If empty OCR, maybe just output empty file.
            pass
            
        if args.dry_run:
            logger.info(f"Dry run: Would translate page {idx}")
            continue
            
        # Translate
        try:
            translation_output = translator.translate_page(body_text, page_info)
        except Exception as e:
            logger.error(f"Failed to translate page {idx}: {e}")
            results.append({'page_index': idx, 'status': 'failed', 'translation_model': args.model, 'notes': str(e)})
            continue
            
        # Metadata updates
        fm['translation_engine'] = "gemini"
        fm['translation_model'] = args.model
        fm['translation_style'] = "meaning_preserving"
        
        translation_notes = []
        confidence = float(page_info.get('mean_confidence', 100))
        if confidence < 60:
            translation_notes.append("LOW OCR CONFIDENCE — review needed")
            
        if translation_notes:
            fm['translation_notes'] = "; ".join(translation_notes)
            
        # Construct Output Content
        
        # En only: Frontmatter + English part
        # We need to extract the English part from the Gemini response
        # Gemini usually returns "## English ... "
        # We allow the model response to be the content, but we want to strip markdown fences if it added them
        
        clean_response = translation_output.strip()
        if clean_response.startswith("```markdown"):
            clean_response = clean_response[11:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        clean_response = clean_response.strip()
        
        # English File
        en_content = "---\n" + yaml.dump(fm, sort_keys=False) + "---\n\n" + clean_response
        
        # Bilingual File
        # Format: Frontmatter + ## Punjabi (OCR) + body + response
        bi_content = "---\n" + yaml.dump(fm, sort_keys=False) + "---\n\n" 
        bi_content += "## Punjabi (OCR)\n" + body_text.strip() + "\n\n"
        bi_content += clean_response
        
        # Write files
        with open(out_en_path, 'w', encoding='utf-8') as f:
            f.write(en_content)
            
        with open(out_bi_path, 'w', encoding='utf-8') as f:
            f.write(bi_content)
            
        results.append({'page_index': idx, 'status': 'success', 'translation_model': args.model, 'notes': ''})
        
        # Sleep slightly to avoid aggressive rate limits even with backoff
        time.sleep(0.5)

    # Write report
    if not args.dry_run:
        file_exists = os.path.exists(out_report_path)
        with open(out_report_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=report_fieldnames)
            if not file_exists:
                writer.writeheader()
            for res in results:
                writer.writerow(res)
    
    logger.info("Translation completed.")

if __name__ == "__main__":
    main()
