"""
Output Writer module for RAG PDF Parser.

Handles multiple output formats with full provenance tracking.
"""

import os
import json
from typing import List, Optional
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.block import IRDocument, IRBlock, IRPage
from src.models.chunk import IRChunk


class OutputWriter:
    """
    Handles multiple output formats for processed documents.

    Supports:
    - Markdown with optional citation anchors
    - JSONL blocks for fine-grained retrieval
    - JSONL chunks for embedding pipelines
    - Metadata JSON for document info
    """

    def __init__(self, base_output_dir: str = "output"):
        """
        Initialize the output writer.

        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = base_output_dir

    def get_doc_output_dir(self, doc_id: str) -> str:
        """Get or create document-specific output directory."""
        doc_dir = os.path.join(self.base_output_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        return doc_dir

    def write_markdown(
        self,
        ir_doc: IRDocument,
        output_dir: Optional[str] = None,
        with_anchors: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Write backward-compatible markdown with optional anchors.

        Args:
            ir_doc: Processed IR document
            output_dir: Output directory (uses doc-specific if not provided)
            with_anchors: Include citation anchors
            filename: Custom filename (default: document.md)

        Returns:
            Path to generated file
        """
        if output_dir is None:
            output_dir = self.get_doc_output_dir(ir_doc.doc_id)
        else:
            os.makedirs(output_dir, exist_ok=True)

        lines = []

        # Document title
        if ir_doc.title:
            lines.append(f"# {ir_doc.title}")
            lines.append("")

        current_page = 0

        # Process all blocks in reading order
        for page in ir_doc.pages:
            # Add page separator for multi-page docs
            if page.page_num != current_page and ir_doc.total_pages > 1:
                if current_page > 0:
                    lines.append("")
                    lines.append(f"---")
                    lines.append(f"<!-- Page {page.page_num} -->")
                    lines.append("")
                current_page = page.page_num

            for block in page.get_blocks_sorted():
                md = self._block_to_markdown(block, with_anchors)
                if md:
                    lines.append(md)
                    lines.append("")

        # Metadata footer
        if with_anchors:
            lines.extend(self._generate_metadata_footer(ir_doc))

        content = "\n".join(lines)

        # Write file
        fname = filename or "document.md"
        output_path = os.path.join(output_dir, fname)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path

    def write_ir_jsonl(
        self,
        ir_doc: IRDocument,
        output_dir: Optional[str] = None,
        filename: str = "blocks.jsonl"
    ) -> str:
        """
        Write IR blocks as JSONL (one block per line).

        Args:
            ir_doc: Processed IR document
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to generated file
        """
        if output_dir is None:
            output_dir = self.get_doc_output_dir(ir_doc.doc_id)
        else:
            os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for page in ir_doc.pages:
                for block in page.blocks:
                    f.write(block.to_json() + "\n")

        return output_path

    def write_chunks_jsonl(
        self,
        chunks: List[IRChunk],
        output_dir: str,
        filename: str = "chunks.jsonl"
    ) -> str:
        """
        Write pre-computed chunks as JSONL.

        Args:
            chunks: List of IR chunks
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to generated file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.to_json() + "\n")

        return output_path

    def write_metadata(
        self,
        ir_doc: IRDocument,
        output_dir: Optional[str] = None,
        filename: str = "metadata.json"
    ) -> str:
        """
        Write document metadata as JSON.

        Args:
            ir_doc: Processed IR document
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to generated file
        """
        if output_dir is None:
            output_dir = self.get_doc_output_dir(ir_doc.doc_id)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Compute statistics
        total_blocks = sum(len(p.blocks) for p in ir_doc.pages)
        block_types = {}
        total_text_chars = 0
        avg_confidence = 0.0
        confidence_count = 0

        for page in ir_doc.pages:
            for block in page.blocks:
                block_types[block.type] = block_types.get(block.type, 0) + 1
                if block.text:
                    total_text_chars += len(block.text)
                if block.confidence > 0:
                    avg_confidence += block.confidence
                    confidence_count += 1

        if confidence_count > 0:
            avg_confidence /= confidence_count

        metadata = {
            "doc_id": ir_doc.doc_id,
            "source_path": ir_doc.source_path,
            "filename": ir_doc.filename,
            "title": ir_doc.title,
            "total_pages": ir_doc.total_pages,
            "total_blocks": total_blocks,
            "block_types": block_types,
            "total_text_chars": total_text_chars,
            "avg_confidence": round(avg_confidence, 4),
            "created_at": ir_doc.created_at,
            "parser_version": ir_doc.parser_version,
            "authors": ir_doc.authors
        }

        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return output_path

    def write_all(
        self,
        ir_doc: IRDocument,
        chunks: Optional[List[IRChunk]] = None,
        output_dir: Optional[str] = None,
        with_anchors: bool = True
    ) -> dict:
        """
        Write all output formats.

        Args:
            ir_doc: Processed IR document
            chunks: Optional pre-computed chunks
            output_dir: Output directory
            with_anchors: Include citation anchors in markdown

        Returns:
            Dictionary of output paths
        """
        if output_dir is None:
            output_dir = self.get_doc_output_dir(ir_doc.doc_id)

        paths = {
            "markdown": self.write_markdown(ir_doc, output_dir, with_anchors),
            "blocks_jsonl": self.write_ir_jsonl(ir_doc, output_dir),
            "metadata": self.write_metadata(ir_doc, output_dir)
        }

        if chunks:
            paths["chunks_jsonl"] = self.write_chunks_jsonl(chunks, output_dir)

        return paths

    def _block_to_markdown(self, block: IRBlock, with_anchors: bool = False) -> str:
        """
        Convert a single block to markdown.

        Args:
            block: IRBlock to convert
            with_anchors: Include citation anchor

        Returns:
            Markdown string
        """
        md = block.markdown or block.text or ""

        # Handle visual blocks without markdown
        if not md and block.has_visual_content():
            if block.image_path:
                md = f"![{block.type}]({block.image_path})"
            else:
                md = f"[{block.type.title()}]"

            if block.caption:
                md += f"\n\n*{block.caption}*"

        # Add anchor if requested
        if with_anchors and block.anchor and md:
            md = f"{md} {block.anchor}"

        return md

    def _generate_metadata_footer(self, ir_doc: IRDocument) -> List[str]:
        """Generate metadata footer for markdown."""
        lines = [
            "",
            "---",
            "",
            "## Document Metadata",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Document ID | `{ir_doc.doc_id}` |",
            f"| Pages | {ir_doc.total_pages} |",
            f"| Processed | {ir_doc.created_at} |",
            f"| Parser Version | {ir_doc.parser_version} |",
        ]

        if ir_doc.title:
            lines.append(f"| Title | {ir_doc.title} |")

        return lines
