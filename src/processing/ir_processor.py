"""
IR Pipeline Processor - Core processing engine for RAG PDF Parser.

Replaces MarkdownAggregator with full IR (Intermediate Representation) support.
Preserves all metadata through processing stages for provenance tracking.
"""

import os
import hashlib
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import fitz  # PyMuPDF

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.block import IRBlock, IRPage, IRDocument
from src.config import ProcessorConfig
from src.layout.detector import LayoutDetector
from src.text.extractor import TextExtractor
from src.captioning.vlm import ImageCaptioner
from src.table.extractor import TableExtractor


class IRPipelineProcessor:
    """
    IR-based document processing pipeline.

    Processes PDF documents while preserving full metadata for RAG applications.
    """

    # Block types that contain visual content
    VISUAL_TYPES = {'figure', 'picture', 'chart', 'table', 'formula'}

    # Block types that contain text content
    TEXT_TYPES = {'text', 'title', 'section_header', 'caption', 'footnote', 'header', 'footer'}

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize the IR Pipeline Processor.

        Args:
            config: Processing configuration. Uses defaults if not provided.
        """
        self.config = config or ProcessorConfig()

        print("Initializing IR Pipeline Processor...")
        print(f"  - Output mode: {self.config.output_mode}")
        print(f"  - DPI: {self.config.dpi}")

        # Initialize components
        self.layout_detector = LayoutDetector()
        self.text_extractor = TextExtractor(lang=self.config.ocr_lang)
        self.captioner = ImageCaptioner(
            model=self.config.vlm_model,
            host=self.config.ollama_host,
            max_concurrent=self.config.vlm_concurrency
        )
        self.table_extractor = TableExtractor()

        # Optional components
        self.translator = None
        self.deduplicator = None

        if self.config.enable_translation:
            from src.translation.translator import Translator
            self.translator = Translator(
                model=self.config.translation_model,
                host=self.config.ollama_host
            )

        if self.config.enable_dedup:
            from src.dedup.deduplicator import Deduplicator
            self.deduplicator = Deduplicator(db_path=self.config.dedup_db_path)

        # Processing state
        self.current_section = None

    def process_document(self, doc: fitz.Document, doc_id: Optional[str] = None) -> IRDocument:
        """
        Process an entire PDF document.

        Args:
            doc: PyMuPDF Document object
            doc_id: Optional document ID. Generated from content if not provided.

        Returns:
            IRDocument with all pages and blocks processed
        """
        # Generate doc_id from file content if not provided
        if doc_id is None:
            # Read first page to generate hash
            if doc.page_count > 0:
                page = doc[0]
                pix = page.get_pixmap(dpi=72)
                doc_id = hashlib.sha256(pix.samples).hexdigest()[:16]
            else:
                doc_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]

        # Get source path and filename
        source_path = doc.name if hasattr(doc, 'name') else ""
        filename = os.path.basename(source_path) if source_path else f"doc_{doc_id}"

        print(f"Processing document: {filename} ({doc.page_count} pages)")

        # Create IR document
        ir_doc = IRDocument(
            doc_id=doc_id,
            source_path=source_path,
            filename=filename,
            total_pages=doc.page_count,
            pages=[],
            parser_version="2.0.0"
        )

        # Process each page
        for page_idx in range(doc.page_count):
            print(f"  Processing page {page_idx + 1}/{doc.page_count}...")
            page = doc[page_idx]
            ir_page = self.process_page(page, page_idx + 1, doc_id)
            ir_doc.pages.append(ir_page)

        # Extract document title from first title block
        for page in ir_doc.pages:
            for block in page.blocks:
                if block.type == 'title' and block.text:
                    ir_doc.title = block.text.strip()
                    break
            if ir_doc.title:
                break

        return ir_doc

    def process_page(
        self,
        page: fitz.Page,
        page_num: int,
        doc_id: str
    ) -> IRPage:
        """
        Process a single page.

        Args:
            page: PyMuPDF Page object
            page_num: 1-indexed page number
            doc_id: Parent document ID

        Returns:
            IRPage with all blocks processed
        """
        # Render page to image
        pix = page.get_pixmap(dpi=self.config.dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Create IR page
        ir_page = IRPage(
            doc_id=doc_id,
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
            dpi=self.config.dpi,
            blocks=[]
        )

        # Detect layout and get IR blocks
        ir_blocks = self.layout_detector.detect_with_metadata(
            image, page_num, doc_id, self.config.dpi
        )

        if not ir_blocks:
            print(f"    Warning: No blocks detected on page {page_num}, using fallback OCR")
            # Fallback: Full page OCR
            text, lines, confidence = self.text_extractor.extract_text_with_metadata(
                image, [0, 0, image.width, image.height]
            )
            fallback_block = IRBlock(
                doc_id=doc_id,
                page=page_num,
                block_id=f"p{page_num}_b0",
                type="text",
                bbox=[0, 0, image.width, image.height],
                reading_order=0,
                text=text,
                markdown=text,
                lang=self.text_extractor.detect_language(text),
                confidence=confidence,
                source_hash="",
                ocr_lines=lines
            )
            ir_page.blocks.append(fallback_block)
            return ir_page

        # Separate text and visual blocks
        text_blocks = [b for b in ir_blocks if b.type in self.TEXT_TYPES]
        visual_blocks = [b for b in ir_blocks if b.type in self.VISUAL_TYPES]

        # Batch OCR for text blocks
        if text_blocks:
            text_bboxes = [b.bbox for b in text_blocks]
            ocr_results = self.text_extractor.extract_text_batch(image, text_bboxes)

            for block, (text, lines, confidence) in zip(text_blocks, ocr_results):
                block.text = text
                block.ocr_lines = lines
                block.confidence = confidence
                block.lang = self.text_extractor.detect_language(text)

                # Format as markdown based on type
                block.markdown = self._format_text_markdown(block)

        # Process visual blocks (with VLM captioning)
        if visual_blocks:
            self._process_visual_blocks(image, visual_blocks, page_num, doc_id)

        # Combine and sort all blocks
        all_blocks = text_blocks + visual_blocks
        all_blocks.sort(key=lambda b: b.reading_order)

        # Track section headers
        for block in all_blocks:
            if block.type in ('title', 'section_header'):
                self.current_section = block.text

        ir_page.blocks = all_blocks

        return ir_page

    def _process_visual_blocks(
        self,
        image: Image.Image,
        blocks: List[IRBlock],
        page_num: int,
        doc_id: str
    ):
        """
        Process visual blocks with VLM captioning.

        Args:
            image: Source page image
            blocks: List of visual IRBlocks to process
            page_num: Current page number
            doc_id: Document ID
        """
        # Prepare crop images for batch captioning
        crop_images = []
        valid_blocks = []

        for block in blocks:
            x1, y1, x2, y2 = [int(v) for v in block.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crop_images.append(crop)
                valid_blocks.append(block)

        if not crop_images:
            return

        # Batch caption generation
        captions = self.captioner.caption_batch(crop_images)

        # Assign captions to blocks
        for block, caption in zip(valid_blocks, captions):
            block.caption = caption

            # For tables, also try structured extraction
            if block.type == 'table':
                table_md = self.table_extractor.extract_table(
                    image, block.bbox
                )
                if table_md and '|' in table_md:
                    block.markdown = f"{table_md}\n\n*AI Summary: {caption}*"
                else:
                    block.markdown = f"[Table]\n\n*AI Summary: {caption}*"
            else:
                # Format figure/chart markdown
                block.markdown = self._format_visual_markdown(block)

    def _format_text_markdown(self, block: IRBlock) -> str:
        """
        Format text block as markdown.

        Args:
            block: IRBlock with text content

        Returns:
            Formatted markdown string
        """
        text = block.text or ""

        if block.type == 'title':
            return f"# {text}"
        elif block.type == 'section_header':
            return f"## {text}"
        elif block.type == 'caption':
            return f"*{text}*"
        elif block.type == 'footnote':
            return f"[^{block.reading_order}]: {text}"
        else:
            return text

    def _format_visual_markdown(self, block: IRBlock) -> str:
        """
        Format visual block as markdown.

        Args:
            block: IRBlock with visual content

        Returns:
            Formatted markdown string with caption
        """
        anchor = block.anchor
        caption = block.caption or "No description available"

        if block.type == 'chart':
            return f"[Chart] {anchor}\n\n*{caption}*"
        elif block.type == 'formula':
            return f"[Formula] {anchor}\n\n*{caption}*"
        else:  # figure, picture
            if block.image_path:
                return f"![{block.type}]({block.image_path})\n\n*{caption}*"
            return f"[Figure] {anchor}\n\n*{caption}*"

    def export_markdown(
        self,
        ir_doc: IRDocument,
        output_dir: str,
        with_anchors: bool = False
    ) -> str:
        """
        Export IR document to markdown format.

        Args:
            ir_doc: Processed IRDocument
            output_dir: Output directory
            with_anchors: Include citation anchors

        Returns:
            Generated markdown content
        """
        lines = []

        # Document header
        if ir_doc.title:
            lines.append(f"# {ir_doc.title}")
            lines.append("")

        # Process all pages
        for page in ir_doc.pages:
            for block in page.get_blocks_sorted():
                md = block.markdown or block.text or ""

                if with_anchors and block.anchor:
                    md = f"{md} {block.anchor}"

                if md:
                    lines.append(md)
                    lines.append("")

        # Add metadata footer if anchors enabled
        if with_anchors:
            lines.append("---")
            lines.append("")
            lines.append("## Document Metadata")
            lines.append("")
            lines.append(f"- **Document ID**: {ir_doc.doc_id}")
            lines.append(f"- **Pages**: {ir_doc.total_pages}")
            lines.append(f"- **Processed**: {ir_doc.created_at}")
            lines.append(f"- **Parser Version**: {ir_doc.parser_version}")

        content = "\n".join(lines)

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ir_doc.doc_id}.md")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    def export_jsonl(self, ir_doc: IRDocument, output_dir: str) -> str:
        """
        Export IR blocks to JSONL format.

        Args:
            ir_doc: Processed IRDocument
            output_dir: Output directory

        Returns:
            Path to generated JSONL file
        """
        import json

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ir_doc.doc_id}_blocks.jsonl")

        with open(output_path, 'w', encoding='utf-8') as f:
            for page in ir_doc.pages:
                for block in page.blocks:
                    f.write(block.to_json() + "\n")

        return output_path

    def save_images(
        self,
        ir_doc: IRDocument,
        doc: fitz.Document,
        output_dir: str
    ):
        """
        Save cropped images for visual blocks.

        Args:
            ir_doc: Processed IRDocument
            doc: Original PyMuPDF Document
            output_dir: Output directory for images
        """
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        for page in ir_doc.pages:
            # Render page
            pix = doc[page.page_num - 1].get_pixmap(dpi=self.config.dpi)
            page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            for block in page.blocks:
                if block.type in self.VISUAL_TYPES:
                    # Crop and save image
                    x1, y1, x2, y2 = [int(v) for v in block.bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(page_image.width, x2)
                    y2 = min(page_image.height, y2)

                    if x2 > x1 and y2 > y1:
                        crop = page_image.crop((x1, y1, x2, y2))
                        filename = f"{block.block_id}_{block.source_hash[:8]}.png"
                        filepath = os.path.join(images_dir, filename)
                        crop.save(filepath)
                        block.image_path = f"images/{filename}"


def process_pdf_file(
    pdf_path: str,
    output_dir: str,
    config: Optional[ProcessorConfig] = None
) -> IRDocument:
    """
    Convenience function to process a PDF file.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        config: Optional processing configuration

    Returns:
        Processed IRDocument
    """
    config = config or ProcessorConfig()
    processor = IRPipelineProcessor(config)

    # Generate doc_id from file content
    doc_id = IRDocument.generate_doc_id(pdf_path)

    # Open and process document
    doc = fitz.open(pdf_path)
    ir_doc = processor.process_document(doc, doc_id)

    # Save outputs based on config
    if config.output_mode in ("markdown", "both"):
        processor.export_markdown(ir_doc, output_dir, config.with_anchors)

    if config.output_mode in ("jsonl", "both"):
        processor.export_jsonl(ir_doc, output_dir)

    # Save images
    processor.save_images(ir_doc, doc, output_dir)

    doc.close()

    return ir_doc
