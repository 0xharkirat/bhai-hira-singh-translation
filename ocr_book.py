
import argparse
import csv
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, ImageOps
import pandas as pd

# --- Engine Abstraction ---
# Try to import OCR libraries and alert the user if they are missing.
try:
    from paddleocr import PaddleOCR
    _PADDLE_INSTALLED = True
except ImportError:
    _PADDLE_INSTALLED = False

try:
    import pytesseract
    # Check if tesseract executable is available
    pytesseract.get_tesseract_version()
    _TESSERACT_INSTALLED = True
except (ImportError, pytesseract.TesseractNotFoundError):
    _TESSERACT_INSTALLED = False

try:
    import easyocr
    _EASYOCR_INSTALLED = True
except ImportError:
    _EASYOCR_INSTALLED = False

# --- Helper Functions ---

def get_sorted_images(input_dir):
    """Gets all image files from a directory, sorted alphabetically."""
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    files = [p for p in Path(input_dir).glob('*') if p.suffix.lower() in supported_exts]
    return sorted(files)

def get_engine_availability():
    """Returns a dictionary of available OCR engines."""
    return {
        "paddle": _PADDLE_INSTALLED,
        "tesseract": _TESSERACT_INSTALLED,
        "easyocr": _EASYOCR_INSTALLED,
    }

def clean_text(text):
    """Basic text cleanup."""
    text = text.replace('  ', ' ')
    text = text.replace('\n ', '\n')
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    return text

# --- OCR Engine Wrapper Classes ---

class BaseOCREngine:
    """Abstract base class for OCR engines."""
    def __init__(self, lang='pa'):
        self.lang = lang
        self.name = "base"

    def ocr(self, image_path_or_np, region=None):
        """Performs OCR on an image. Returns text, mean confidence, and word count."""
        raise NotImplementedError

class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCR engine wrapper."""
    def __init__(self, lang='pan'): # Tesseract uses 'pan' for Punjabi
        super().__init__(lang)
        self.name = "tesseract"
        if not _TESSERACT_INSTALLED:
            raise ImportError("Tesseract is not installed or not in PATH.")
        # Check for language data
        if self.lang not in pytesseract.get_languages(config='') :
            print(f"Warning: Tesseract language pack '{self.lang}' not found.", file=sys.stderr)

    def ocr(self, image_np, region=None):
        img_to_ocr = image_np.copy()
        if region:
            x, y, w, h = region
            img_to_ocr = img_to_ocr[y:y+h, x:x+w]

        try:
            ocr_data = pytesseract.image_to_data(img_to_ocr, lang=self.lang, output_type=pytesseract.Output.DATAFRAME)
            ocr_data = ocr_data[ocr_data.conf != -1]
            
            if ocr_data.empty:
                return "", 0, 0

            mean_conf = ocr_data['conf'].mean()
            word_count = len(ocr_data['text'].dropna())
            text = pytesseract.image_to_string(img_to_ocr, lang=self.lang)
            
            return clean_text(text), mean_conf, word_count
        except Exception as e:
            print(f"Error during Tesseract OCR: {e}", file=sys.stderr)
            return "", 0, 0

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine wrapper."""
    def __init__(self, lang='pa'):
        super().__init__(lang)
        self.name = "paddle"
        if not _PADDLE_INSTALLED:
            raise ImportError("paddleocr or paddlepaddle is not installed.")
        # PaddleOCR uses 'pa' for Punjabi, which is part of its multilingual model
        self.engine = PaddleOCR(use_angle_cls=True, lang='pa', show_log=False)

    def ocr(self, image_path_or_np, region=None):
        # PaddleOCR prefers file paths for some optimizations
        img_to_ocr = image_path_or_np
        if region:
            x, y, w, h = region
            if isinstance(image_path_or_np, (str, Path)):
                img_to_ocr = cv2.imread(str(image_path_or_np))
            img_to_ocr = img_to_ocr[y:y+h, x:x+w]
        
        try:
            result = self.engine.ocr(img_to_ocr, cls=True)
            if not result or not result[0]:
                return "", 0, 0

            lines = result[0]
            text_lines = [line[1][0] for line in lines]
            confidences = [line[1][1] for line in lines]
            
            text = "\n".join(text_lines)
            mean_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0
            word_count = sum(len(line.split()) for line in text_lines)

            return clean_text(text), mean_conf, word_count
        except Exception as e:
            print(f"Error during PaddleOCR: {e}", file=sys.stderr)
            return "", 0, 0


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine wrapper."""
    def __init__(self, lang='pa'):
        super().__init__(lang)
        self.name = "easyocr"
        if not _EASYOCR_INSTALLED:
            raise ImportError("easyocr is not installed.")
        self.engine = easyocr.Reader(['pa'], gpu=False) # Specify Punjabi, disable GPU for simplicity

    def ocr(self, image_np, region=None):
        img_to_ocr = image_np.copy()
        if region:
            x, y, w, h = region
            img_to_ocr = img_to_ocr[y:y+h, x:x+w]
        
        try:
            result = self.engine.readtext(img_to_ocr, paragraph=True)
            if not result:
                return "", 0, 0
            
            text_blocks = [res[1] for res in result]
            confidences = [res[2] for res in result]

            text = "\n\n".join(text_blocks)
            mean_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0
            word_count = sum(len(block.split()) for block in text_blocks)

            return clean_text(text), mean_conf, word_count
        except Exception as e:
            print(f"Error during EasyOCR: {e}", file=sys.stderr)
            return "", 0, 0


def get_ocr_engine(engine_choice='auto'):
    """Factory function to get the best available OCR engine."""
    availability = get_engine_availability()
    
    if engine_choice == 'auto':
        if availability['paddle']:
            return PaddleOCREngine()
        if availability['tesseract']:
            return TesseractOCREngine()
        if availability['easyocr']:
            return EasyOCREngine()
        return None
    elif engine_choice == 'paddle':
        if availability['paddle']: return PaddleOCREngine()
        else: raise RuntimeError("PaddleOCR was requested but is not installed.")
    elif engine_choice == 'tesseract':
        if availability['tesseract']: return TesseractOCREngine()
        else: raise RuntimeError("Tesseract was requested but is not installed.")
    elif engine_choice == 'easyocr':
        if availability['easyocr']: return EasyOCREngine()
        else: raise RuntimeError("EasyOCR was requested but is not installed.")
    
    return None

# --- Image Processing and Classification ---

class PageProcessor:
    """Handles preprocessing, classification, and OCR strategy for a single page."""
    def __init__(self, debug=False, debug_dir=None):
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None

    def _save_debug_image(self, image_np, filename_suffix):
        if self.debug and self.debug_dir:
            try:
                debug_path = self.debug_dir / f"{self.base_filename}_{filename_suffix}.jpg"
                cv2.imwrite(str(debug_path), image_np)
            except Exception as e:
                print(f"Failed to save debug image: {e}", file=sys.stderr)

    def preprocess(self, image_path):
        """Loads and preprocesses an image for OCR."""
        self.base_filename = image_path.stem

        # 1. Load image respecting EXIF orientation
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img_np = np.array(img.convert('RGB'))
        except Exception as e:
            print(f"Could not read image {image_path}: {e}", file=sys.stderr)
            return None
        
        # 2. Convert to Grayscale for most operations
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 3. Denoise lightly
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        self._save_debug_image(denoised, "01_denoised")

        # 4. Attempt auto-rotation (simple version using Tesseract OSD)
        # More complex Hough transform is an option but often less reliable.
        # PaddleOCR has its own rotation correction, so we only do this for Tesseract.
        notes = []
        if _TESSERACT_INSTALLED:
            try:
                osd = pytesseract.image_to_osd(denoised, output_type=pytesseract.Output.DICT)
                rotation = osd['rotate']
                if rotation != 0:
                    (h, w) = denoised.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
                    denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    notes.append(f"rotated {-rotation} degrees")
                    self._save_debug_image(denoised, "02_rotated")
            except Exception:
                pass # OSD can fail on non-text images

        # 5. Binarize
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        self._save_debug_image(binary, "03_binary")

        return binary, notes

    def classify_page(self, binary_image):
        """Classifies page using OpenCV heuristics."""
        h, w = binary_image.shape
        
        # Invert image so text is white on black background
        inverted = cv2.bitwise_not(binary_image)
        
        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "COVER_OR_SEPARATOR", None

        # Heuristics:
        # - FIGURE_PAGE: one or few very large contours (the photo itself).
        # - TEXT_PAGE: many small to medium sized contours (words/lines).
        # - COVER_OR_SEPARATOR: very few contours, large white space.
        
        contour_areas = [cv2.contourArea(c) for c in contours]
        total_contour_area = sum(contour_areas)
        
        # Normalize by page area
        text_area_ratio = total_contour_area / (w * h)
        
        # Check for large "photo-like" contours
        large_contour_threshold = 0.3 * (w * h)
        has_large_contour = any(area > large_contour_threshold for area in contour_areas)

        if has_large_contour and text_area_ratio < 0.5:
            page_type = "FIGURE_PAGE"
        elif text_area_ratio > 0.08: # High density of text
            page_type = "TEXT_PAGE"
        elif 0.01 < text_area_ratio <= 0.08: # Could be a clipping or a sparse text page
            page_type = "CLIPPING_PAGE"
        else: # Very low text density
            page_type = "COVER_OR_SEPARATOR"

        # Refine for FIGURE_PAGE: find the likely caption area at the bottom
        caption_region = None
        if page_type == "FIGURE_PAGE":
            # Assume caption is in the bottom 20% of the page
            caption_y_start = int(h * 0.80)
            caption_region = (0, caption_y_start, w, h - caption_y_start)
            
            # Save debug crop
            if self.debug:
                debug_img_crop = binary_image.copy()
                cv2.rectangle(debug_img_crop, (caption_region[0], caption_region[1]), 
                              (caption_region[0]+caption_region[2], caption_region[1]+caption_region[3]), 
                              (127), 3)
                self._save_debug_image(debug_img_crop, "04_caption_detect")

        return page_type, caption_region

# --- Main Application Logic ---

def main():
    parser = argparse.ArgumentParser(description="OCR a book from images.")
    parser.add_argument("--input", required=True, help="Directory with input images.")
    parser.add_argument("--out", required=True, help="Directory for output files.")
    parser.add_argument("--engine", default="auto", choices=["auto", "paddle", "tesseract", "easyocr"],
                        help="Force a specific OCR engine.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing files.")
    parser.add_argument("--start", type=int, default=1, help="Page number to start processing from.")
    parser.add_argument("--end", type=int, default=None, help="Page number to stop processing at.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images to debug folder.")
    args = parser.parse_args()

    # --- Setup ---
    input_dir = Path(args.input)
    output_dir = Path(args.out)
    pa_output_dir = output_dir / "pa"
    debug_dir = output_dir / "debug"

    print("Setting up output directories...")
    pa_output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    # --- Engine Selection ---
    print(f"Engine mode: {args.engine}")
    print(f"Available engines: {get_engine_availability()}")
    try:
        ocr_engine = get_ocr_engine(args.engine)
        if ocr_engine is None:
            print("No suitable OCR engine found. Please install PaddleOCR or Tesseract.", file=sys.stderr)
            sys.exit(1)
        print(f"Selected engine: {ocr_engine.name}")
    except (RuntimeError, ImportError) as e:
        print(f"Error initializing OCR engine: {e}", file=sys.stderr)
        sys.exit(1)


    # --- File Processing ---
    print(f"Finding images in {input_dir}...")
    images = get_sorted_images(input_dir)
    
    # Handle toc.jpg separately first
    toc_image = Path(input_dir) / "toc.jpg"
    if toc_image in images:
        images.remove(toc_image)
        images.insert(0, toc_image)

    total_images = len(images)
    print(f"Found {total_images} images.")
    
    # Slice for start/end
    start_index = args.start - 1
    end_index = args.end if args.end is not None else total_images
    images_to_process = images[start_index:end_index]

    if not images_to_process:
        print("No images to process in the specified range.")
        return

    # --- Processing Loop ---
    page_processor = PageProcessor(debug=args.debug, debug_dir=debug_dir)
    all_results = []
    
    print(f"Processing {len(images_to_process)} pages (from {args.start} to {end_index})...")
    for i, image_path in enumerate(images_to_process):
        page_index = start_index + i + 1
        
        is_toc = image_path.name == "toc.jpg"
        output_filename = "toc.pa.md" if is_toc else f"page_{page_index:04d}.pa.md"
        output_path = pa_output_dir / output_filename

        print(f"[{page_index}/{total_images}] Processing {image_path.name} -> {output_path.name}")

        if output_path.exists() and not args.force:
            print("  -> Output exists, skipping (use --force to override).")
            # To generate a full report, we need to read the existing file
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # A simple way to parse frontmatter without extra deps
                if content.startswith('---'):
                    frontmatter_str = content.split('---')[1]
                    data = {}
                    for line in frontmatter_str.strip().split('\n'):
                        key, val = line.split(':', 1)
                        data[key.strip()] = val.strip().strip('"')
                    all_results.append(data)
            continue
        
        # 1. Preprocess
        processed_image, notes = page_processor.preprocess(image_path)
        if processed_image is None:
            continue
            
        # 2. Classify
        page_type, region_to_ocr = page_processor.classify_page(processed_image)
        
        # 3. OCR
        ocr_notes = list(notes)
        text, mean_confidence, word_count = "", 0, 0
        
        if page_type == "FIGURE_PAGE":
            ocr_notes.append("caption-only")
            if region_to_ocr:
                # For Paddle, pass the full path; for others, pass the numpy array
                img_input = str(image_path) if ocr_engine.name == 'paddle' else processed_image
                text, mean_confidence, word_count = ocr_engine.ocr(img_input, region=region_to_ocr)
        elif page_type in ["TEXT_PAGE", "CLIPPING_PAGE", "COVER_OR_SEPARATOR"]:
             # For Paddle, pass the full path; for others, pass the numpy array
            img_input = str(image_path) if ocr_engine.name == 'paddle' else processed_image
            text, mean_confidence, word_count = ocr_engine.ocr(img_input)

        # 4. Confidence Check
        low_confidence = mean_confidence < 70
        # Suspiciously low word count for text pages
        if page_type in ["TEXT_PAGE", "CLIPPING_PAGE"] and word_count < 20:
            low_confidence = True
            ocr_notes.append("low word count for text page")

        # 5. Assemble and Write Output
        result_data = {
            "source_image": image_path.name,
            "page_index": page_index,
            "page_type": page_type,
            "engine_used": ocr_engine.name,
            "mean_confidence": round(mean_confidence, 2),
            "low_confidence": str(low_confidence).lower(), # Convert boolean to string "true" or "false"
            "word_count": word_count,
            "notes": ", ".join(ocr_notes),
        }
        all_results.append(result_data)

        frontmatter = "---\n"
        for key, value in result_data.items():
            frontmatter += f'{key}: "{value}"\n'
        frontmatter += "---\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter)
            f.write(text)

    # --- Final Report Generation ---
    print("Generating final reports...")
    
    # report.csv
    report_path = output_dir / "report.csv"
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by="page_index")
        # Reorder columns for clarity
        cols = ["page_index", "source_image", "page_type", "engine_used", "mean_confidence", "low_confidence", "word_count", "notes"]
        df = df[cols]
        df.to_csv(report_path, index=False)
        print(f"  -> Saved {report_path}")

    # manifest.json
    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "run_timestamp": datetime.now().isoformat(),
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "processed_files": all_results,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  -> Saved {manifest_path}")

    # qa_summary.md
    qa_path = output_dir / "qa_summary.md"
    with open(qa_path, 'w', encoding='utf-8') as f:
        f.write("# OCR QA Summary\n\n")
        
        low_conf_pages = [r for r in all_results if r['low_confidence']]
        if low_conf_pages:
            f.write("## Pages with Low Confidence (<70% or low word count)\n\n")
            for page in low_conf_pages:
                f.write(f"- **Page {page['page_index']}** ({page['source_image']}): Confidence {page['mean_confidence']}%, Words: {page['word_count']}\n")
            f.write("\n")

        figure_pages = [r for r in all_results if r['page_type'] == 'FIGURE_PAGE']
        if figure_pages:
            f.write("## Pages Classified as Figures (Caption-Only OCR)\n\n")
            for page in figure_pages:
                f.write(f"- **Page {page['page_index']}** ({page['source_image']})\n")
            f.write("\n")
        
        # Note on fallback is harder without re-implementing engine logic here
        # But we can show if a non-default was used.
        if ocr_engine.name != 'paddle' and _PADDLE_INSTALLED:
             f.write("## Engine Fallback Notes\n\n")
             f.write(f"The preferred engine 'paddle' was available but '{ocr_engine.name}' was used.\n")
             f.write("This may be because of the `--engine` flag or an automatic fallback.")

    print(f"  -> Saved {qa_path}")
    print("\nOCR process complete.")


if __name__ == "__main__":
    main()
