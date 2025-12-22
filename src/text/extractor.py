"""
Text Extraction module using PaddleOCR.

Extracts text from document images with column-aware sorting.
Enhanced to support IR with coordinate mapping and confidence tracking.
"""

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import hashlib


class TextExtractor:
    """
    PaddleOCR-based text extractor with IR support.
    """

    def __init__(self, lang: str = 'korean'):
        """
        Initialize PaddleOCR Text Extractor.

        Args:
            lang: OCR language ('korean', 'en', 'ch', etc.)
        """
        print(f"Loading Text Extractor (PaddleOCR) [lang={lang}]...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.lang = lang

    def extract_text(self, image: Image.Image, bbox: Optional[List[float]] = None) -> Tuple[str, List[Dict]]:
        """
        Extract text from an image or a specific bbox within the image.

        Args:
            image: PIL Image to extract text from
            bbox: Optional [x1, y1, x2, y2] to crop before OCR

        Returns:
            Tuple of (full_text, lines) where lines contain text, confidence, and box info
        """
        if bbox:
            img_w, img_h = image.size
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            crop_img = image.crop((x1, y1, x2, y2))
        else:
            crop_img = image

        img_np = np.array(crop_img)
        result = self.ocr.ocr(img_np)

        full_text = ""
        lines = []

        if result and len(result) > 0:
            res_item = result[0]
            if isinstance(res_item, dict):
                lines = self._parse_dict_result(res_item, image.width if image else 1000)
            elif isinstance(res_item, list):
                lines = self._parse_list_result(res_item, image.width if image else 1000)

            for line in lines:
                full_text += line['text'] + " "

        return full_text.strip(), lines

    def extract_text_with_metadata(
        self,
        image: Image.Image,
        bbox: List[float],
        page_coords: bool = True
    ) -> Tuple[str, List[Dict], float]:
        """
        Extract text with full metadata including page-absolute coordinates.

        Args:
            image: Source PIL Image
            bbox: [x1, y1, x2, y2] region to extract
            page_coords: If True, convert line coordinates to page-absolute

        Returns:
            Tuple of (text, lines_with_page_coords, average_confidence)
        """
        img_w, img_h = image.size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return "", [], 0.0

        crop_img = image.crop((x1, y1, x2, y2))
        img_np = np.array(crop_img)

        result = self.ocr.ocr(img_np)

        full_text = ""
        lines = []
        total_confidence = 0.0

        if result and len(result) > 0:
            res_item = result[0]
            if isinstance(res_item, dict):
                raw_lines = self._parse_dict_result(res_item, crop_img.width)
            elif isinstance(res_item, list):
                raw_lines = self._parse_list_result(res_item, crop_img.width)
            else:
                raw_lines = []

            # Convert to page coordinates if requested
            if page_coords:
                lines = self._map_coords_to_page(raw_lines, x1, y1)
            else:
                lines = raw_lines

            for line in lines:
                full_text += line['text'] + " "
                total_confidence += line.get('confidence', 0.0)

        avg_confidence = total_confidence / len(lines) if lines else 0.0

        return full_text.strip(), lines, avg_confidence

    def extract_text_batch(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        return_confidence: bool = True
    ) -> List[Tuple[str, List[Dict], float]]:
        """
        Batch extract text from multiple bboxes with full metadata.

        Args:
            image: Source PIL Image
            bboxes: List of [x1, y1, x2, y2] regions
            return_confidence: Include average confidence in results

        Returns:
            List of (text, lines, avg_confidence) tuples in same order as bboxes
        """
        if not bboxes:
            return []

        img_w, img_h = image.size

        # Prepare crops and track their original indices
        crops = []
        crop_offsets = []  # (x_offset, y_offset) for coordinate mapping

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(np.array(crop))
                crop_offsets.append((x1, y1))
            else:
                crops.append(None)
                crop_offsets.append((0, 0))

        # Filter valid crops for batch OCR
        valid_indices = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_indices]

        # Batch OCR call
        if valid_crops:
            batch_results = self.ocr.ocr(valid_crops)
        else:
            batch_results = []

        # Map results back to original order with page coordinates
        results = [("", [], 0.0) for _ in bboxes]

        for idx, result_idx in enumerate(valid_indices):
            if idx < len(batch_results) and batch_results[idx]:
                text, raw_lines = self._parse_ocr_result(batch_results[idx], image.width)

                # Map to page coordinates
                x_offset, y_offset = crop_offsets[result_idx]
                lines = self._map_coords_to_page(raw_lines, x_offset, y_offset)

                # Calculate average confidence
                avg_conf = 0.0
                if lines:
                    avg_conf = sum(l.get('confidence', 0.0) for l in lines) / len(lines)

                results[result_idx] = (text, lines, avg_conf)

        return results

    def _map_coords_to_page(
        self,
        lines: List[Dict],
        x_offset: float,
        y_offset: float
    ) -> List[Dict]:
        """
        Map crop-relative coordinates to page-absolute coordinates.

        Args:
            lines: List of line dicts with 'box' field
            x_offset: X offset of crop region in page
            y_offset: Y offset of crop region in page

        Returns:
            Lines with page-absolute coordinates
        """
        mapped_lines = []

        for line in lines:
            mapped_line = line.copy()
            box = line.get('box', [])

            if box:
                # Box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                mapped_box = []
                for point in box:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        mapped_box.append([
                            point[0] + x_offset,
                            point[1] + y_offset
                        ])
                    else:
                        mapped_box.append(point)
                mapped_line['box'] = mapped_box
                mapped_line['box_page_absolute'] = True

            mapped_lines.append(mapped_line)

        return mapped_lines

    def _parse_dict_result(self, res_item: Dict, image_width: int) -> List[Dict]:
        """Parse dictionary-style OCR result (PaddleX/v3 format)."""
        d_texts = res_item.get('rec_texts', [])
        d_scores = res_item.get('rec_scores', [])
        d_boxes = res_item.get('rec_boxes', [])

        if len(d_boxes) == 0:
            d_boxes = res_item.get('dt_polys', res_item.get('dt_boxes', []))

        raw_lines = []
        count = min(len(d_texts), len(d_boxes)) if d_boxes else len(d_texts)

        for i in range(count):
            raw_lines.append({
                'text': d_texts[i],
                'confidence': d_scores[i] if i < len(d_scores) else 0.0,
                'box': d_boxes[i] if i < len(d_boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            })

        return self.sort_boxes(raw_lines, image_width)

    def _parse_list_result(self, res_item: List, image_width: int) -> List[Dict]:
        """Parse list-style OCR result (legacy format)."""
        raw_lines = []

        for line in res_item:
            coord = line[0]
            try:
                text, conf = line[1]
            except (ValueError, IndexError):
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                conf = 0.0

            raw_lines.append({
                'text': text,
                'confidence': conf,
                'box': coord
            })

        return self.sort_boxes(raw_lines, image_width)

    def _parse_ocr_result(self, result, image_width: int) -> Tuple[str, List[Dict]]:
        """Parse single OCR result into text and lines."""
        full_text = ""
        lines = []

        if not result:
            return full_text, lines

        res_item = result[0] if isinstance(result, list) and result else result

        if isinstance(res_item, dict):
            lines = self._parse_dict_result(res_item, image_width)
        elif isinstance(res_item, list):
            lines = self._parse_list_result(res_item, image_width)

        for line in lines:
            full_text += line['text'] + " "

        return full_text.strip(), lines

    def sort_boxes(self, lines: List[Dict], image_width: int) -> List[Dict]:
        """
        Sort text boxes in reading order with column awareness.

        Args:
            lines: List of line dicts with 'box' field
            image_width: Width of the source image

        Returns:
            Sorted list of lines
        """
        if not lines:
            return []

        # Normalize box format
        def get_points(box):
            if len(box) == 4 and isinstance(box[0], (int, float, np.number)):
                x1, y1, x2, y2 = box
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            return box

        for line in lines:
            line['box'] = get_points(line['box'])

        def get_center(box):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        # Column detection for multi-column layouts
        if len(lines) > 10:
            mid_x = image_width / 2
            left_items = [l for l in lines if get_center(l['box'])[0] < mid_x]
            right_items = [l for l in lines if get_center(l['box'])[0] >= mid_x]

            if left_items and right_items:
                y_min_l = min([min([p[1] for p in l['box']]) for l in left_items])
                y_max_l = max([max([p[1] for p in l['box']]) for l in left_items])
                y_min_r = min([min([p[1] for p in l['box']]) for l in right_items])
                y_max_r = max([max([p[1] for p in l['box']]) for l in right_items])

                overlap = max(0, min(y_max_l, y_max_r) - max(y_min_l, y_min_r))
                total_h = max(y_max_l, y_max_r) - min(y_min_l, y_min_r)

                if total_h > 0 and (overlap / total_h) > 0.3:
                    return self.sort_boxes(left_items, mid_x) + self.sort_boxes(right_items, image_width - mid_x)

        # Standard sort: top-down, left-right
        def sort_key(line):
            c = get_center(line['box'])
            return (int(c[1] / 10) * 10, c[0])

        return sorted(lines, key=sort_key)

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character distribution.

        Args:
            text: Text to analyze

        Returns:
            Language code ('ko', 'en', 'zh', 'unknown')
        """
        if not text:
            return "unknown"

        # Count character types
        korean = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())

        total = korean + chinese + ascii_chars
        if total == 0:
            return "unknown"

        if korean / total > 0.3:
            return "ko"
        elif chinese / total > 0.3:
            return "zh"
        elif ascii_chars / total > 0.5:
            return "en"

        return "unknown"
