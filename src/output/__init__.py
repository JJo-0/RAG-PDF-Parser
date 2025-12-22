"""
Output module for RAG PDF Parser.

Handles multiple output formats: Markdown, JSONL, and chunks.
"""

from .writer import OutputWriter

__all__ = ['OutputWriter']
