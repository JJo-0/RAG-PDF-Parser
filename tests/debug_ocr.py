import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text.extractor import TextExtractor
import fitz
from PIL import Image
import numpy as np

def debug_ocr_page():
    pdf_path = "tests/Predictive Modeling of Human Behavior During Exoskeleton Assisted Walking.pdf"
    print(f"Loading {pdf_path}...")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    print("Initializing TextExtractor...")
    extractor = TextExtractor()
    
    print("Extracting text from full page...")
    # Add temporary print in checking Loop or just print result here
    # We will rely on the printed logs inside extractor if we modify it, 
    # but here let's just inspect the returned structure if we can.
    
    # To really see 'line', we need to peek into the class or use the one I modified (which prints on error).
    # I'll modify the extractor to ALWAYS print the first line structure it sees.
    
    text, lines = extractor.extract_text(img)
    print("--- Extracted Text Preview ---")
    print(text[:200])
    print("------------------------------")
    print(f"Lines detected: {len(lines)}")
    if lines:
        print("First line object:", lines[0])

if __name__ == "__main__":
    debug_ocr_page()
