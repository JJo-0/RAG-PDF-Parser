"""
Configuration module for RAG PDF Parser.

Centralizes all configuration options for the processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from argparse import Namespace


@dataclass
class ProcessorConfig:
    """
    Main configuration for the IR Pipeline Processor.

    Controls output modes, chunking, translation, and performance settings.
    """
    # Output settings
    output_mode: str = "markdown"        # "markdown" | "jsonl" | "both"
    with_anchors: bool = False           # Add citation anchors to markdown
    output_dir: str = "output"           # Base output directory

    # Chunking settings
    enable_chunking: bool = False        # Generate pre-chunked output
    chunk_size: int = 1000               # Target chunk size in tokens
    chunk_overlap: int = 100             # Overlap between chunks
    respect_sections: bool = True        # Keep section headers with content

    # Translation settings
    enable_translation: bool = False     # Include translation in output
    source_lang: Optional[str] = None    # Auto-detect if None
    target_lang: str = "en"              # Target language for translation
    translation_model: str = "gpt-oss:20b"  # Ollama translation model
    bilingual_output: bool = False       # Include both original and translation

    # Deduplication settings
    enable_dedup: bool = False           # Skip duplicate documents
    dedup_db_path: str = "output/.dedup_db.json"

    # Page merging settings
    merge_pages: bool = False            # Merge split sentences across page boundaries
    merge_model: str = "qwen3:32b-q4_K_M"  # Model for page merge detection
    merge_context_chars: int = 500       # Context chars around page boundary

    # VLM/Captioning settings
    vlm_model: str = "qwen3-vl:8b"       # VLM model for captioning
    vlm_concurrency: int = 3             # Max concurrent VLM requests
    structured_captions: bool = True     # Use structured caption prompts
    caption_context_chars: int = 200     # Surrounding text for caption context

    # OCR settings
    ocr_lang: str = "korean"             # PaddleOCR language
    ocr_batch_size: int = 10             # Max regions per OCR batch

    # Rendering settings
    dpi: int = 200                       # PDF rendering DPI
    max_image_dimension: int = 2048      # Max image dimension for VLM

    # Performance settings
    enable_cache: bool = True            # Use persistent cache
    cache_db_path: str = "output/.cache.db"
    gpu_memory_limit_mb: int = 8000      # GPU memory limit

    # Ollama settings
    ollama_host: str = "http://localhost:11434"

    @classmethod
    def from_args(cls, args: Namespace) -> 'ProcessorConfig':
        """Create configuration from argparse namespace."""
        config = cls()

        # Map argparse attributes to config fields
        arg_mapping = {
            'output_mode': 'output_mode',
            'with_anchors': 'with_anchors',
            'output_dir': 'output_dir',
            'chunk': 'enable_chunking',
            'chunk_size': 'chunk_size',
            'translate': 'enable_translation',
            'target_lang': 'target_lang',
            'bilingual': 'bilingual_output',
            'dedup': 'enable_dedup',
            'merge_pages': 'merge_pages',
            'dpi': 'dpi',
            'vlm_model': 'vlm_model',
        }

        for arg_name, config_name in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(config, config_name, value)

        return config

    @classmethod
    def for_fast_processing(cls) -> 'ProcessorConfig':
        """Preset for fast processing with reduced quality."""
        return cls(
            dpi=150,
            vlm_concurrency=5,
            structured_captions=False,
            enable_cache=True,
            chunk_size=1500
        )

    @classmethod
    def for_high_quality(cls) -> 'ProcessorConfig':
        """Preset for high quality processing."""
        return cls(
            dpi=300,
            vlm_concurrency=2,
            structured_captions=True,
            enable_cache=True,
            chunk_size=800,
            with_anchors=True
        )

    @classmethod
    def for_rag_pipeline(cls) -> 'ProcessorConfig':
        """Preset optimized for RAG embedding pipeline."""
        return cls(
            output_mode="both",
            with_anchors=True,
            enable_chunking=True,
            chunk_size=1000,
            chunk_overlap=100,
            structured_captions=True,
            enable_cache=True
        )


@dataclass
class LayoutConfig:
    """Configuration for layout detection."""
    model_name: str = "vikparuchuri/surya_layout2"
    confidence_threshold: float = 0.5
    detect_reading_order: bool = True
    detect_tables: bool = True


@dataclass
class OCRConfig:
    """Configuration for OCR extraction."""
    lang: str = "korean"                 # "korean", "en", "ch"
    use_gpu: bool = True
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    use_angle_cls: bool = True
    batch_size: int = 10


@dataclass
class VLMConfig:
    """Configuration for VLM captioning."""
    model: str = "qwen3-vl:8b"
    host: str = "http://localhost:11434"
    max_concurrent: int = 3
    timeout: int = 60
    temperature: float = 0.3
    max_tokens: int = 500

    # Prompt templates
    figure_prompt: str = """Analyze this figure/image. Respond in JSON format only:
{"type": "diagram|photo|illustration|screenshot", "title": "visible or inferred title", "description": "concise description (max 100 words)", "key_elements": ["element1", "element2"], "text_in_image": ["any visible text"]}"""

    chart_prompt: str = """Analyze this chart/graph. Respond in JSON format only:
{"chart_type": "bar|line|pie|scatter|other", "title": "chart title", "x_axis": {"label": "", "range": ""}, "y_axis": {"label": "", "range": ""}, "data_summary": "key trends", "legend_items": []}"""

    table_prompt: str = """Analyze this table image. Respond in JSON format only:
{"title": "table title if visible", "columns": ["col1", "col2"], "row_count": 0, "summary": "what this table shows"}"""

    generic_prompt: str = """Describe this image in detail. If it contains a chart, explain the data trends. If it contains a diagram, explain the components and their relationships."""


# Global default configuration
DEFAULT_CONFIG = ProcessorConfig()
