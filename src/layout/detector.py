"""
Layout Detection module using Surya.

Detects document layout regions including text, titles, figures, tables, etc.
Enhanced to support IR (Intermediate Representation) with full metadata.
"""

from surya.layout import LayoutPredictor, FoundationPredictor
from PIL import Image
import numpy as np
import torch
import hashlib
from typing import List, Optional, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.block import IRBlock


class LayoutDetector:
    """
    Surya-based layout detector with IR support.
    """

    # Map Surya labels to normalized IR types
    LABEL_MAP = {
        'text': 'text',
        'title': 'title',
        'section-header': 'section_header',
        'section_header': 'section_header',
        'sectionheader': 'section_header',
        'picture': 'figure',
        'figure': 'figure',
        'image': 'figure',
        'table': 'table',
        'chart': 'chart',
        'formula': 'formula',
        'equation': 'formula',
        'caption': 'caption',
        'footnote': 'footnote',
        'list-item': 'text',
        'page-header': 'header',
        'page-footer': 'footer',
    }

    def __init__(self, model_path: str = "vikparuchuri/surya_layout2"):
        """
        Initialize Surya Layout Detector.

        Args:
            model_path: Path or HuggingFace model ID for Surya layout model
        """
        print(f"Loading Layout Detector model: {model_path}...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Using device: {self.device}")

        self.foundation = FoundationPredictor(device=self.device)
        self.model = LayoutPredictor(self.foundation)
        self.model_path = model_path

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect layout regions in the image.

        Returns a list of blocks with label, bbox, and reading order.
        Backward compatible with existing code.

        Args:
            image: PIL Image to analyze

        Returns:
            List of block dictionaries with 'label', 'bbox', 'order' keys
        """
        results = self.model([image])
        result = results[0]

        blocks = []

        if hasattr(result, 'bboxes'):
            for idx, block in enumerate(result.bboxes):
                order = getattr(block, 'order', None)
                if order is None:
                    order = idx  # Fallback to list index

                blocks.append({
                    "label": block.label,
                    "bbox": list(block.bbox) if hasattr(block.bbox, '__iter__') else block.bbox,
                    "order": order,
                    "confidence": getattr(block, 'confidence', getattr(block, 'score', 0.0))
                })

        elif hasattr(result, 'layout'):
            for idx, block in enumerate(result.layout):
                order = getattr(block, 'order', None)
                if order is None:
                    order = idx

                blocks.append({
                    "label": block.label,
                    "bbox": list(block.bbox) if hasattr(block.bbox, '__iter__') else block.bbox,
                    "order": order,
                    "confidence": getattr(block, 'confidence', getattr(block, 'score', 0.0))
                })
        else:
            print(f"Warning: Unknown Surya result format. Keys: {dir(result)}")

        # Sort by reading order
        blocks.sort(key=lambda x: x.get('order', 0))

        return blocks

    def detect_with_metadata(
        self,
        image: Image.Image,
        page_num: int,
        doc_id: str,
        dpi: int = 200
    ) -> List[IRBlock]:
        """
        Enhanced detection returning IRBlock objects with full metadata.

        Args:
            image: PIL Image to analyze
            page_num: 1-indexed page number
            doc_id: Document ID for block ID generation
            dpi: DPI used for rendering (for coordinate normalization)

        Returns:
            List of IRBlock objects with full provenance metadata
        """
        raw_blocks = self.detect(image)
        ir_blocks = []

        for idx, block in enumerate(raw_blocks):
            order = block.get('order', idx)
            label = block.get('label', 'unknown').lower()
            normalized_type = self.LABEL_MAP.get(label, 'text')

            # Generate stable block ID
            block_id = f"p{page_num}_b{order}"

            # Generate source hash from bbox region
            bbox = block.get('bbox', [0, 0, 0, 0])
            source_hash = self._compute_region_hash(image, bbox)

            # Create IRBlock
            ir_block = IRBlock(
                doc_id=doc_id,
                page=page_num,
                block_id=block_id,
                type=normalized_type,
                bbox=list(bbox),
                reading_order=order,
                text=None,  # To be filled by OCR
                markdown=None,
                lang="unknown",
                confidence=block.get('confidence', 0.0),
                source_hash=source_hash,
                caption=None,
                translation=None,
                image_path=None
            )

            ir_blocks.append(ir_block)

        return ir_blocks

    def _compute_region_hash(self, image: Image.Image, bbox: List[float]) -> str:
        """
        Compute hash of a cropped region for deduplication.

        Args:
            image: Source image
            bbox: [x1, y1, x2, y2] coordinates

        Returns:
            SHA256 hash (first 12 chars) of the region
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                return ""

            crop = image.crop((x1, y1, x2, y2))
            # Resize for consistent hashing
            crop_small = crop.resize((64, 64), Image.Resampling.LANCZOS)
            pixels = np.array(crop_small).tobytes()
            return hashlib.sha256(pixels).hexdigest()[:12]

        except Exception:
            return ""

    def get_visual_blocks(self, blocks: List[IRBlock]) -> List[IRBlock]:
        """
        Filter blocks that contain visual content (figures, tables, charts).

        Args:
            blocks: List of IRBlock objects

        Returns:
            Filtered list of visual content blocks
        """
        visual_types = {'figure', 'table', 'chart', 'formula', 'picture'}
        return [b for b in blocks if b.type.lower() in visual_types]

    def get_text_blocks(self, blocks: List[IRBlock]) -> List[IRBlock]:
        """
        Filter blocks that contain text content.

        Args:
            blocks: List of IRBlock objects

        Returns:
            Filtered list of text content blocks
        """
        text_types = {'text', 'title', 'section_header', 'caption', 'footnote'}
        return [b for b in blocks if b.type.lower() in text_types]
