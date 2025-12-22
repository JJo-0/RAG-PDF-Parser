"""
Cache module for RAG PDF Parser.

Provides persistent caching for OCR, layout, and VLM results.
"""

from .persistent import PersistentCache

__all__ = ['PersistentCache']
