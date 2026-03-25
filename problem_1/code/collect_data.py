"""
CSL 7640: Natural Language Understanding - Assignment 2
Script   : collect_data.py
Purpose  : Read all files from the dataset/ folder (PDFs and TXTs),
           extract raw text from each, and save into raw_corpus.txt
           NO cleaning is done here — just raw text extraction.
Author   : Nirmal Kumar Godara
"""

import os
import pdfplumber  # used to extract text from PDF files

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Folder where all your source files are kept
DATASET_DIR = "dataset"

# Output file — raw text from all sources combined
OUTPUT_FILE = "raw_corpus.txt"


# ─────────────────────────────────────────────
# FUNCTION: Extract text from a PDF file
# ─────────────────────────────────────────────

def extract_from_pdf(filepath):
    """
    Opens a PDF file and extracts text from every page.
    Uses pdfplumber which handles complex layouts well.
    Returns all extracted text as a single string.
    """
    all_text = []

    with pdfplumber.open(filepath) as pdf:
        print(f"    Pages found: {len(pdf.pages)}")
        for page in pdf.pages:
            page_text = page.extract_text()  # extract text from one page
            if page_text:                    # some pages may be images (no text)
                all_text.append(page_text)

    return "\n".join(all_text)  # join all pages with newline


# ─────────────────────────────────────────────
# FUNCTION: Extract text from a TXT file
# ─────────────────────────────────────────────

def extract_from_txt(filepath):
    """
    Simply reads a .txt file and returns its content.
    Uses utf-8 encoding, ignores any unreadable characters.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ─────────────────────────────────────────────
# MAIN: Loop through dataset/ folder
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 50)
    print("collect_data.py — Raw Text Extraction")
    print("CSL 7640 - Assignment 2, Task 1")
    print("=" * 50)

    all_raw_text = []   # will hold text from all files
    total_files  = 0    # counter for how many files processed

    # Loop through every file in the dataset/ folder
    for filename in os.listdir(DATASET_DIR):
        filepath = os.path.join(DATASET_DIR, filename)

        # ── Handle PDF files ──
        if filename.endswith(".pdf"):
            print(f"\n  [PDF] Reading: {filename}")
            try:
                text = extract_from_pdf(filepath)
                all_raw_text.append(f"\n\n--- SOURCE: {filename} ---\n\n{text}")
                print(f"    Extracted {len(text.split())} words")
                total_files += 1
            except Exception as e:
                print(f"    [ERROR] Could not read {filename}: {e}")

        # ── Handle TXT files ──
        elif filename.endswith(".txt"):
            print(f"\n  [TXT] Reading: {filename}")
            try:
                text = extract_from_txt(filepath)
                all_raw_text.append(f"\n\n--- SOURCE: {filename} ---\n\n{text}")
                print(f"    Extracted {len(text.split())} words")
                total_files += 1
            except Exception as e:
                print(f"    [ERROR] Could not read {filename}: {e}")

        # ── Skip unknown file types ──
        else:
            print(f"\n  [SKIP] Unsupported file type: {filename}")

    # Save all extracted text into raw_corpus.txt

    final_raw_text = "\n".join(all_raw_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_raw_text)


    # Print summary

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Total files processed : {total_files}")
    print(f"  Total words extracted : {len(final_raw_text.split())}")
    print(f"  Raw corpus saved to   : {OUTPUT_FILE}")
    print("=" * 50)
    print("\nNext step: run preprocess.py")