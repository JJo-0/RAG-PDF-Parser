"""
Core IR Block data models for the RAG PDF Parser.

These models preserve metadata through all processing stages:
- Layout detection (Surya)
- OCR extraction (PaddleOCR)
- VLM captioning (Ollama)
- Translation
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json


@dataclass
class IRBlock:
    """
    Intermediate Representation of a single document block.

    Preserves all metadata for provenance tracking and RAG retrieval.
    """
    # Core identifiers
    doc_id: str                          # SHA256[:16] of source PDF
    page: int                            # 1-indexed page number
    block_id: str                        # Format: "p{page}_b{order}"

    # Type and position
    type: str                            # "text"|"title"|"section_header"|"table"|"figure"|"chart"|"formula"
    bbox: List[float]                    # [x1, y1, x2, y2] in page coordinates
    reading_order: int                   # Surya reading order (or computed)

    # Content
    text: Optional[str] = None           # OCR extracted text
    markdown: Optional[str] = None       # Formatted markdown representation

    # Quality metrics
    lang: str = "unknown"                # Detected language code ("en", "ko")
    confidence: float = 0.0              # OCR/detection confidence [0.0-1.0]
    source_hash: str = ""                # SHA256 of cropped image bytes

    # Parser metadata
    parser_source: str = "unknown"       # Parser that generated this block ("ppstructure"|"qwenvl")

    # Enrichment fields
    caption: Optional[str] = None        # VLM-generated caption for figures/charts
    caption_structured: Optional[Dict[str, Any]] = None  # Structured VLM output
    translation: Optional[str] = None    # Translated text
    image_path: Optional[str] = None     # Relative path to saved image crop

    # Citation anchor
    anchor: str = ""                     # Format: "[@p{page}_{type}{order}]"

    # OCR line-level details (for fine-grained retrieval)
    ocr_lines: Optional[List[Dict[str, Any]]] = None  # [{text, box, confidence}]

    # Raw data from PPStructureV3 (for tables, includes HTML structure)
    raw_data: Optional[Dict[str, Any]] = None  # e.g., {'structure': '<table>...</table>', 'bbox_list': [...]}

    def __post_init__(self):
        """Generate anchor if not provided."""
        if not self.anchor:
            type_abbrev = {
                'text': 'txt', 'title': 'ttl', 'section_header': 'sec',
                'table': 'tbl', 'figure': 'fig', 'chart': 'cht',
                'formula': 'eq', 'picture': 'fig'
            }.get(self.type.lower(), self.type[:3])
            self.anchor = f"[@p{self.page}_{type_abbrev}{self.reading_order}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def convert_value(obj):
            """Convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, (np.integer, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_value(value) for key, value in obj.items()}
            return obj

        data = asdict(self)
        # Convert all values to native Python types
        return {key: convert_value(value) for key, value in data.items()}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRBlock':
        """Create IRBlock from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'IRBlock':
        """Create IRBlock from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_content(self) -> str:
        """Get the best available text content."""
        return self.markdown or self.text or self.caption or ""

    def has_visual_content(self) -> bool:
        """Check if this block contains visual content (image/table/chart)."""
        return self.type.lower() in ('figure', 'picture', 'chart', 'table', 'formula')


@dataclass
class IRPage:
    """
    Intermediate Representation of a single document page.
    """
    doc_id: str
    page_num: int                        # 1-indexed
    width: float                         # Page width in points
    height: float                        # Page height in points
    dpi: int                             # Rendering DPI used
    blocks: List[IRBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_id': self.doc_id,
            'page_num': self.page_num,
            'width': self.width,
            'height': self.height,
            'dpi': self.dpi,
            'blocks': [b.to_dict() for b in self.blocks]
        }

    def get_blocks_by_type(self, block_type: str) -> List[IRBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.type.lower() == block_type.lower()]

    def get_blocks_sorted(self) -> List[IRBlock]:
        """Get blocks sorted by reading order."""
        # Handle None values in reading_order (e.g., headers/footers may not have order)
        return sorted(self.blocks, key=lambda b: b.reading_order if b.reading_order is not None else 999)


@dataclass
class IRDocument:
    """
    Intermediate Representation of an entire document.
    """
    doc_id: str                          # SHA256[:16] of source PDF
    source_path: str                     # Original file path
    filename: str                        # Original filename
    total_pages: int
    pages: List[IRPage] = field(default_factory=list)
    created_at: str = ""                 # ISO timestamp
    parser_version: str = "2.0.0"        # Parser version for compatibility

    # Optional metadata
    title: Optional[str] = None
    authors: Optional[List[str]] = None

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_id': self.doc_id,
            'source_path': self.source_path,
            'filename': self.filename,
            'total_pages': self.total_pages,
            'pages': [p.to_dict() for p in self.pages],
            'created_at': self.created_at,
            'parser_version': self.parser_version,
            'title': self.title,
            'authors': self.authors
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRDocument':
        """Create IRDocument from dictionary."""
        pages_data = data.pop('pages', [])
        doc = cls(**{k: v for k, v in data.items() if k != 'pages'})

        for page_data in pages_data:
            blocks_data = page_data.pop('blocks', [])
            page = IRPage(**page_data)
            page.blocks = [IRBlock.from_dict(b) for b in blocks_data]
            doc.pages.append(page)

        return doc

    def all_blocks(self) -> List[IRBlock]:
        """Get all blocks from all pages in reading order."""
        all_blocks = []
        for page in self.pages:
            all_blocks.extend(page.get_blocks_sorted())
        return all_blocks

    def get_block_by_id(self, block_id: str) -> Optional[IRBlock]:
        """Find a block by its ID."""
        for page in self.pages:
            for block in page.blocks:
                if block.block_id == block_id:
                    return block
        return None

    def get_blocks_by_anchor(self, anchor: str) -> Optional[IRBlock]:
        """Find a block by its citation anchor."""
        for page in self.pages:
            for block in page.blocks:
                if block.anchor == anchor:
                    return block
        return None

    @staticmethod
    def generate_doc_id(file_path: str) -> str:
        """Generate document ID from file content."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
