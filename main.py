"""
RAG PDF Parser - Main Entry Point

Processes PDF documents for RAG applications with full metadata preservation.
"""

import os
import argparse
import fitz  # PyMuPDF
from warnings import filterwarnings

# Import IR pipeline components
from src.processing.ir_processor import IRPipelineProcessor, process_pdf_file
from src.processing.chunking import chunk_ir_document
from src.processing.page_merger import PageMerger
from src.output.writer import OutputWriter
from src.config import ProcessorConfig
from src.models.chunk import ChunkingConfig

# Suppress warnings
filterwarnings("ignore")


def process_file(pdf_path: str, config: ProcessorConfig, processor: IRPipelineProcessor):
    """
    Process a single PDF file with the IR pipeline.

    Args:
        pdf_path: Path to PDF file
        config: Processing configuration
        processor: IR Pipeline Processor instance
    """
    from src.models.block import IRDocument

    base_name = os.path.basename(pdf_path)
    file_name = os.path.splitext(base_name)[0]

    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")

    # Check for duplicates if enabled
    if config.enable_dedup and processor.deduplicator:
        existing = processor.deduplicator.check_pdf(pdf_path)
        if existing:
            print(f"  [SKIP] Duplicate detected: {existing.get('filename', 'unknown')}")
            return

    # Generate document ID
    doc_id = IRDocument.generate_doc_id(pdf_path)
    print(f"  Document ID: {doc_id}")

    # Open and process document
    doc = fitz.open(pdf_path)
    ir_doc = processor.process_document(doc, doc_id)

    # Create output writer
    writer = OutputWriter(config.output_dir)

    # Determine output directory
    if config.output_mode == "markdown":
        # Legacy mode: output directly to output_dir
        output_dir = config.output_dir
    else:
        # New mode: create doc-specific directory
        output_dir = os.path.join(config.output_dir, doc_id)

    # Save images
    print(f"  Saving images...")
    processor.save_images(ir_doc, doc, output_dir)

    # Generate outputs based on mode
    if config.output_mode in ("markdown", "both"):
        print(f"  Generating Markdown...")
        if config.output_mode == "markdown":
            # Default filename format
            md_filename = f"{file_name}_qwenvl.md"
            md_path = writer.write_markdown(
                ir_doc, output_dir, config.with_anchors,
                filename=md_filename
            )
        else:
            md_path = writer.write_markdown(ir_doc, output_dir, config.with_anchors)
        print(f"    -> {md_path}")

        # Merge pages if requested
        if config.merge_pages:
            print(f"  Merging split sentences across pages...")
            merger = PageMerger(
                model=config.merge_model,
                host=config.ollama_host,
                context_chars=config.merge_context_chars
            )
            merged_path = merger.process_file(md_path)
            print(f"    -> {merged_path}")

    if config.output_mode in ("jsonl", "both"):
        print(f"  Generating JSONL blocks...")
        jsonl_path = writer.write_ir_jsonl(ir_doc, output_dir)
        print(f"    -> {jsonl_path}")

        print(f"  Generating metadata...")
        meta_path = writer.write_metadata(ir_doc, output_dir)
        print(f"    -> {meta_path}")

    # Generate chunks if requested
    if config.enable_chunking:
        print(f"  Generating chunks (size={config.chunk_size})...")
        chunk_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            overlap_tokens=config.chunk_overlap,
            respect_sections=config.respect_sections
        )
        chunks = chunk_ir_document(ir_doc, chunk_config)
        chunks_path = writer.write_chunks_jsonl(chunks, output_dir)
        print(f"    -> {chunks_path} ({len(chunks)} chunks)")

    # Register in dedup database if enabled
    if config.enable_dedup and processor.deduplicator:
        processor.deduplicator.register_pdf(pdf_path, ir_doc.total_pages)

    doc.close()

    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(
        description="RAG PDF Parser - Extract structured content from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic markdown output (legacy mode)
  python main.py document.pdf

  # Full IR output with chunks
  python main.py document.pdf --output_mode both --chunk

  # Merge split sentences across pages
  python main.py document.pdf --merge_pages

  # With translation
  python main.py document.pdf --translate --target_lang ko

  # Process directory
  python main.py ./papers/ --output_mode jsonl --chunk
        """
    )

    # Input/Output
    parser.add_argument("input_path", type=str,
                        help="Path to PDF file or directory")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory (default: output)")

    # Output mode
    parser.add_argument("--output_mode", type=str,
                        choices=["markdown", "jsonl", "both"],
                        default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--with_anchors", action="store_true",
                        help="Add citation anchors to markdown")

    # Chunking
    parser.add_argument("--chunk", action="store_true",
                        help="Generate pre-chunked output for embedding")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Target chunk size in tokens (default: 1000)")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Overlap between chunks (default: 100)")

    # Translation
    parser.add_argument("--translate", action="store_true",
                        help="Include translation in output")
    parser.add_argument("--target_lang", type=str, default="en",
                        help="Translation target language (default: en)")
    parser.add_argument("--bilingual", action="store_true",
                        help="Output both original and translation")

    # Deduplication
    parser.add_argument("--dedup", action="store_true",
                        help="Skip duplicate documents")

    # Page merging
    parser.add_argument("--merge_pages", action="store_true",
                        help="Merge split sentences across page boundaries")

    # Processing options
    parser.add_argument("--dpi", type=int, default=200,
                        help="PDF rendering DPI (default: 200)")
    parser.add_argument("--vlm_model", type=str, default="qwen3-vl:8b",
                        help="VLM model for captioning (default: qwen3-vl:8b)")

    args = parser.parse_args()

    # Build configuration
    config = ProcessorConfig(
        output_mode=args.output_mode,
        with_anchors=args.with_anchors,
        output_dir=args.output_dir,
        enable_chunking=args.chunk,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_translation=args.translate,
        target_lang=args.target_lang,
        bilingual_output=args.bilingual,
        enable_dedup=args.dedup,
        merge_pages=args.merge_pages,
        dpi=args.dpi,
        vlm_model=args.vlm_model
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize processor
    print("Initializing RAG PDF Parser...")
    processor = IRPipelineProcessor(config)

    # Process input
    if os.path.isdir(args.input_path):
        # Process all PDFs in directory
        files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.pdf')]
        print(f"\nFound {len(files)} PDF files in {args.input_path}")

        for f in files:
            full_path = os.path.join(args.input_path, f)
            try:
                process_file(full_path, config, processor)
            except Exception as e:
                print(f"  [ERROR] Failed to process {f}: {e}")

    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith('.pdf'):
        process_file(args.input_path, config, processor)

    else:
        print("Error: Invalid input. Please provide a PDF file or directory.")
        return 1

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Output directory: {os.path.abspath(config.output_dir)}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
