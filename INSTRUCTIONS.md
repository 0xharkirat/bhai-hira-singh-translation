# Meaning-Preserving Translation with Gemini

This tool translates Punjabi OCR text to English using the Gemini API. It processes files page-by-page to ensure context is maintained without exceeding token limits.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Key**
    Set your Gemini API key in the environment:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

## Usage

Run the script to process the `./out` directory.

```bash
python translate_book_gemini.py --pa ./out/pa --report ./out/report.csv --out ./out
```

### Options

*   `--start N`: Start from page index N (default: 1)
*   `--end M`: Stop at page index M (default: no limit)
*   `--force`: Overwrite existing translation files
*   `--dry-run`: Read files and prompt but do not call API or write files
*   `--model`: Specify Gemini model (default: `gemini-flash-latest`)
*   `--temperature`: Set generation temperature (default: 0.2)

## Output

The script generates:
*   `./out/en/page_XXXX.en.md`: English translation only
*   `./out/bilingual/page_XXXX.md`: Side-by-side Punjabi and English
*   `./out/translate_report.csv`: Summary of the translation process

## Workflow

1.  The script reads `out/report.csv` to check OCR confidence.
2.  It iterates through `out/pa/*.md` files.
3.  It skips files that already have translations in `out/en` (unless `--force` is used).
4.  It sends the text to Gemini with strict instructions for meaning-preserving translation.
5.  It saves the outputs and updates the report.
