import os
import argparse
import fitz  # PyMuPDF
from PIL import Image
from warnings import filterwarnings

# Import our custom modules
from src.processing.aggregator import MarkdownAggregator

# Suppress warnings
filterwarnings("ignore")

def pdf_to_images(pdf_path):
    """
    Convert all pages of a PDF to PIL Images.
    """
    doc = fitz.open(pdf_path)
    images = []
    
    print(f"  - Rendering {len(doc)} pages from PDF...")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200) # 200 DPI is usually good balance
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    doc.close()
    return images

def process_file(pdf_path, output_dir, aggregator):
    """
    Process a single PDF file.
    """
    base_name = os.path.basename(pdf_path)
    file_name = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{file_name}.md")
    
    # Open the PDF document
    doc = fitz.open(pdf_path)

    print(f"[{base_name}] Processing PDF for Markdown generation...")
    markdown_content = aggregator.aggregate(doc, output_dir) # Pass the fitz.Document object

    # Close the document after processing
    doc.close()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"[{base_name}] Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Local RAG PDF Parser")
    parser.add_argument("input_path", type=str, help="Path to PDF file or directory containing PDFs")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save Markdown files")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize generic aggregator (loads models once)
    aggregator = MarkdownAggregator()

    if os.path.isdir(args.input_path):
        # Process all PDFs in directory
        files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.pdf')]
        print(f"Found {len(files)} PDF files in {args.input_path}")
        
        for f in files:
            full_path = os.path.join(args.input_path, f)
            process_file(full_path, args.output_dir, aggregator)
            
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith('.pdf'):
        process_file(args.input_path, args.output_dir, aggregator)
    else:
        print("Error: Invalid input path. Please provide a PDF file or a directory containing PDFs.")

if __name__ == "__main__":
    main()
