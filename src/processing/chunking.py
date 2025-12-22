"""
Chunking module for RAG PDF Parser.

Provides both legacy markdown-based chunking and new IR-aware chunking.
"""

import re
from typing import List, Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.block import IRBlock
from src.models.chunk import IRChunk, ChunkingConfig, estimate_tokens


def chunk_with_ir_awareness(
    ir_blocks: List[IRBlock],
    config: Optional[ChunkingConfig] = None
) -> List[IRChunk]:
    """
    Chunk IR blocks while preserving full metadata and provenance.

    Args:
        ir_blocks: List of IRBlock objects (should be in reading order)
        config: Chunking configuration

    Returns:
        List of IRChunk objects with source tracking
    """
    config = config or ChunkingConfig()
    chunks = []

    if not ir_blocks:
        return chunks

    # Get doc_id from first block
    doc_id = ir_blocks[0].doc_id

    # Group blocks by section
    current_chunk_blocks: List[IRBlock] = []
    current_section: Optional[str] = None
    current_tokens = 0
    chunk_index = 0

    for block in ir_blocks:
        block_text = block.get_content()
        block_tokens = estimate_tokens(block_text, config.token_method)

        # Check for section change
        is_section_header = block.type in ('title', 'section_header')

        if is_section_header and config.respect_sections:
            # Flush current chunk if substantial
            if current_chunk_blocks and current_tokens >= config.min_chunk_size:
                chunk = _create_chunk(
                    current_chunk_blocks, doc_id, chunk_index,
                    current_section, config
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                if config.overlap_tokens > 0:
                    current_chunk_blocks = _get_overlap_blocks(
                        current_chunk_blocks, config.overlap_tokens, config.token_method
                    )
                    current_tokens = sum(
                        estimate_tokens(b.get_content(), config.token_method)
                        for b in current_chunk_blocks
                    )
                else:
                    current_chunk_blocks = []
                    current_tokens = 0

            current_section = block_text.strip()

        # Add block to current chunk
        current_chunk_blocks.append(block)
        current_tokens += block_tokens

        # Check if we need to flush due to size
        if current_tokens >= config.chunk_size:
            # Try to break at paragraph boundary if respecting paragraphs
            if config.respect_paragraphs and not block_text.strip():
                chunk = _create_chunk(
                    current_chunk_blocks, doc_id, chunk_index,
                    current_section, config
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                if config.overlap_tokens > 0:
                    current_chunk_blocks = _get_overlap_blocks(
                        current_chunk_blocks, config.overlap_tokens, config.token_method
                    )
                    current_tokens = sum(
                        estimate_tokens(b.get_content(), config.token_method)
                        for b in current_chunk_blocks
                    )
                else:
                    current_chunk_blocks = []
                    current_tokens = 0

            # Force break if way over max size
            elif current_tokens >= config.max_chunk_size:
                chunk = _create_chunk(
                    current_chunk_blocks, doc_id, chunk_index,
                    current_section, config
                )
                chunks.append(chunk)
                chunk_index += 1

                if config.overlap_tokens > 0:
                    current_chunk_blocks = _get_overlap_blocks(
                        current_chunk_blocks, config.overlap_tokens, config.token_method
                    )
                    current_tokens = sum(
                        estimate_tokens(b.get_content(), config.token_method)
                        for b in current_chunk_blocks
                    )
                else:
                    current_chunk_blocks = []
                    current_tokens = 0

    # Final chunk
    if current_chunk_blocks:
        chunk = _create_chunk(
            current_chunk_blocks, doc_id, chunk_index,
            current_section, config
        )
        chunks.append(chunk)

    return chunks


def _create_chunk(
    blocks: List[IRBlock],
    doc_id: str,
    chunk_index: int,
    section: Optional[str],
    config: ChunkingConfig
) -> IRChunk:
    """Create an IRChunk from a list of blocks."""
    # Merge text content
    texts = []
    for block in blocks:
        content = block.get_content()
        if content:
            texts.append(content)

    merged_text = "\n\n".join(texts)

    # Compute page range
    pages = [b.page for b in blocks]
    page_range = (min(pages), max(pages)) if pages else (0, 0)

    # Collect block IDs and anchors
    block_ids = [b.block_id for b in blocks]
    anchors = [b.anchor for b in blocks if b.anchor]

    # Reading order range
    orders = [b.reading_order for b in blocks]
    order_start = min(orders) if orders else 0
    order_end = max(orders) if orders else 0

    return IRChunk(
        chunk_id=f"{doc_id[:8]}_c{chunk_index}",
        doc_id=doc_id,
        page_range=page_range,
        block_ids=block_ids,
        section=section,
        text=merged_text,
        token_count=estimate_tokens(merged_text, config.token_method),
        reading_order_start=order_start,
        reading_order_end=order_end,
        anchors=anchors if config.include_anchors else []
    )


def _get_overlap_blocks(
    blocks: List[IRBlock],
    overlap_tokens: int,
    token_method: str
) -> List[IRBlock]:
    """Get blocks for overlap from the end of the current chunk."""
    if not blocks:
        return []

    overlap_blocks = []
    tokens = 0

    # Work backwards to get overlap
    for block in reversed(blocks):
        block_tokens = estimate_tokens(block.get_content(), token_method)
        if tokens + block_tokens <= overlap_tokens:
            overlap_blocks.insert(0, block)
            tokens += block_tokens
        else:
            break

    return overlap_blocks


def chunk_ir_document(
    ir_doc,  # IRDocument
    config: Optional[ChunkingConfig] = None
) -> List[IRChunk]:
    """
    Chunk an entire IR document.

    Args:
        ir_doc: IRDocument to chunk
        config: Chunking configuration

    Returns:
        List of IRChunk objects
    """
    all_blocks = ir_doc.all_blocks()
    return chunk_with_ir_awareness(all_blocks, config)


# ============================================================================
# Legacy support - Original markdown-based chunking
# ============================================================================

def chunk_with_layout_awareness(markdown: str, chunk_size: int = 1000) -> List[Dict]:
    """
    Legacy: Chunk markdown content while preserving structure.

    This function is kept for backward compatibility.
    For new code, use chunk_with_ir_awareness() instead.

    Args:
        markdown: Markdown content to chunk
        chunk_size: Target chunk size in tokens

    Returns:
        List of chunk dictionaries with 'content', 'section', 'type' keys
    """
    lines = markdown.split('\n')
    chunks = []
    current_chunk = []
    current_section = "Introduction"
    token_count = 0

    def count_tokens(text):
        return len(text.split())

    for i, line in enumerate(lines):
        # Section Header Detection
        heading_match = re.match(r'^(#+)\s+(.+)$', line)

        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)

            if level <= 2:
                if current_chunk and token_count > 100:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                    current_chunk = []
                    token_count = 0

                current_section = title
                current_chunk.append(line)
                token_count += count_tokens(line)
            else:
                current_chunk.append(line)
                token_count += count_tokens(line)

        elif line.strip().startswith('|'):
            if not (current_chunk and current_chunk[-1].strip().startswith('|')):
                if current_chunk and token_count > 500:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                    current_chunk = []
                    token_count = 0

            current_chunk.append(line)
            token_count += count_tokens(line)

        elif line.strip().startswith('!['):
            current_chunk.append(line)
            token_count += count_tokens(line)

        else:
            current_chunk.append(line)
            token_count += count_tokens(line)

            if token_count >= chunk_size:
                if not line.strip():
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                    current_chunk = []
                    token_count = 0

    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'section': current_section,
            'type': 'text'
        })

    return chunks
