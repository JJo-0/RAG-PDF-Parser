import sys
import os
import fitz
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.layout.detector import LayoutDetector

def analyze_page(page_num=0):
    pdf_path = "tests/Predictive Modeling of Human Behavior During Exoskeleton Assisted Walking.pdf"
    print(f"Analyzing Page {page_num + 1} of {pdf_path}")
    
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    detector = LayoutDetector()
    blocks = detector.detect(img)
    
if __name__ == "__main__":
    pdf_path = "tests/다중 도메인 비전 시스템 기반 제조 환경 안전 모니터링을 위한 동적 3D 작업자 자세 정합기법.pdf"
    doc = fitz.open(pdf_path)
    print(f"Scanning {pdf_path} ({len(doc)} pages)...")
    
    detector = LayoutDetector()
    
    # Check Page 0 and 1
    for page_num in [0, 1]:
        print(f"--- Page {page_num} ---")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        blocks = detector.detect(img)
        print(f"  Found {len(blocks)} blocks:")
        for i, block in enumerate(blocks):
            print(f"    {i}: {block['label']} {block['bbox']}")
