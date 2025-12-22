"""
IR Chunk data model for RAG embedding and retrieval.

Chunks preserve provenance information linking back to source blocks.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any, Set
import json
import hashlib


@dataclass
class IRChunk:
    """
    Intermediate Representation of a document chunk for embedding.

    Designed for RAG pipelines with full provenance tracking.
    """
    # Identifiers
    chunk_id: str                        # Format: "{doc_id[:8]}_c{index}"
    doc_id: str                          # Parent document ID

    # Source tracking
    page_range: Tuple[int, int] = (0, 0)  # (start_page, end_page)
    block_ids: List[str] = field(default_factory=list)  # Source block IDs

    # Content
    section: Optional[str] = None        # Heading/section context
    text: str = ""                       # Merged text content

    # Metrics
    token_count: int = 0                 # Estimated token count
    char_count: int = 0                  # Character count

    # Reading order tracking
    reading_order_start: int = 0
    reading_order_end: int = 0

    # Citation anchors included in this chunk
    anchors: List[str] = field(default_factory=list)

    # Optional enrichment
    translation: Optional[str] = None
    embedding: Optional[List[float]] = None  # Pre-computed embedding vector

    def __post_init__(self):
        """Compute derived fields."""
        if not self.char_count:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d['page_range'] = list(self.page_range)
        # Exclude embedding from normal serialization (too large)
        if 'embedding' in d:
            d['embedding'] = None
        return d

    def to_json(self) -> str:
        """Convert to JSON string (single line for JSONL)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRChunk':
        """Create IRChunk from dictionary."""
        # Convert list back to tuple for page_range
        if 'page_range' in data and isinstance(data['page_range'], list):
            data['page_range'] = tuple(data['page_range'])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'IRChunk':
        """Create IRChunk from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_citation_footer(self) -> str:
        """Generate citation footer for this chunk."""
        pages = f"p.{self.page_range[0]}"
        if self.page_range[1] != self.page_range[0]:
            pages = f"p.{self.page_range[0]}-{self.page_range[1]}"
        return f"[Source: {self.doc_id}, {pages}, blocks: {len(self.block_ids)}]"

    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:12]


def estimate_tokens(text: str, method: str = "simple") -> int:
    """
    Estimate token count for text.

    Args:
        text: Input text
        method: "simple" (whitespace) or "tiktoken" (more accurate)

    Returns:
        Estimated token count
    """
    if method == "simple":
        # Simple estimation: ~4 chars per token for English, ~2 for CJK
        # This is a rough approximation
        import re
        cjk_count = len(re.findall(r'[\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff]', text))
        other_count = len(text) - cjk_count
        return int(cjk_count / 1.5 + other_count / 4)

    elif method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback to simple if tiktoken not available
            return estimate_tokens(text, method="simple")

    return len(text.split())


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    chunk_size: int = 1000               # Target chunk size in tokens
    overlap_tokens: int = 100            # Overlap between chunks
    respect_sections: bool = True        # Keep section headers with content
    respect_paragraphs: bool = True      # Avoid splitting mid-paragraph
    min_chunk_size: int = 100            # Minimum chunk size
    max_chunk_size: int = 2000           # Maximum chunk size
    include_anchors: bool = True         # Include citation anchors in chunks
    token_method: str = "simple"         # Token counting method
