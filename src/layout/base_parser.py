"""
Abstract base class for document parsers.

Provides a common interface for different parsing approaches:
- PPStructureV3 (layout-first with OCR)
- Qwen3-VL (vision-first with VLM)
- Future parsers (GPT-4V, LLaMA-Vision, etc.)
"""

from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import numpy as np
from src.models.block import IRBlock


class BaseDocumentParser(ABC):
    """
    Abstract interface for document parsers.

    All parsers must implement parse_page() to convert a page image
    into a list of IRBlocks with reading order, type, and content.
    """

    @abstractmethod
    def parse_page(
        self,
        page_image: Image.Image,
        page_num: int,
        doc_id: str
    ) -> List[IRBlock]:
        """
        Parse a single page image into IRBlocks.

        Args:
            page_image: PIL Image of the page
            page_num: 1-indexed page number
            doc_id: Document ID (hash)

        Returns:
            List of IRBlocks with:
            - reading_order: Sequential block ordering
            - type: Block classification (text/title/table/figure/etc.)
            - text: Extracted or inferred text content
            - bbox: Optional bounding box [x1, y1, x2, y2]
            - parser_source: Identifier for this parser
        """
        pass

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """
        Identifier for this parser (e.g., 'ppstructure', 'qwenvl').

        Used for:
        - IRBlock.parser_source field
        - Comparison reports
        - User selection via CLI
        """
        pass

    def supports_tables(self) -> bool:
        """
        Whether this parser can extract table structure.

        Returns:
            True if parser can detect and parse tables
        """
        return False

    def supports_bbox(self) -> bool:
        """
        Whether this parser provides bounding boxes.

        Returns:
            True if parser outputs pixel-level bbox coordinates
        """
        return False

    def supports_formulas(self) -> bool:
        """
        Whether this parser can extract mathematical formulas.

        Returns:
            True if parser can detect and parse formulas (LaTeX)
        """
        return False
