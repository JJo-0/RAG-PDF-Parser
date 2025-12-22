"""
Data models for RAG PDF Parser Intermediate Representation (IR).
"""

from .block import IRBlock, IRPage, IRDocument
from .chunk import IRChunk

__all__ = ['IRBlock', 'IRPage', 'IRDocument', 'IRChunk']
