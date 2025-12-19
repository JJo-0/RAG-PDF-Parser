from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

class TextExtractor:
    def __init__(self, lang='korean'):
        """
        Initialize PaddleOCR Text Extractor.
        """
        print(f"Loading Text Extractor (PaddleOCR) [lang={lang}]...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract_text(self, image: Image.Image, bbox=None):
        """
        Extract text from an image or a specific bbox within the image.
        bbox format: [x1, y1, x2, y2]
        """
        if bbox:
            # Crop the image to the bbox
            # Ensure bbox is within image bounds
            img_w, img_h = image.size
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)
            
            crop_img = image.crop((x1, y1, x2, y2))
        else:
            crop_img = image

        # Convert to numpy array for PaddleOCR
        img_np = np.array(crop_img)
        
        # PaddleOCR expects BGR for cv2, but it handles RGB numpy arrays too usually.
        # Safest is to rely on its internal handling or standard numpy.
        
        result = self.ocr.ocr(img_np)
        
        full_text = ""
        lines = []
        
        if result and len(result) > 0:
            res_item = result[0]
            if isinstance(res_item, dict):
                # New structure (PaddleX / v3 pipeline style?)
                # Keys: rec_texts, rec_scores, rec_boxes (or dt_boxes)
                # We need to assume these are lists of same length
                d_texts = res_item.get('rec_texts', [])
                d_scores = res_item.get('rec_scores', [])
                d_boxes = res_item.get('rec_boxes', [])
                
                # If rec_boxes is missing, maybe dt_polys or dt_boxes
                if len(d_boxes) == 0: 
                    d_boxes = res_item.get('dt_polys', res_item.get('dt_boxes', []))
                
                if len(d_boxes) == 0: 
                    d_boxes = res_item.get('dt_polys', res_item.get('dt_boxes', []))
                
                # Combine into a list of dicts for sorting
                raw_lines = []
                count = min(len(d_texts), len(d_boxes))
                for i in range(count):
                    raw_lines.append({
                        'text': d_texts[i],
                        'confidence': d_scores[i] if i < len(d_scores) else 0.0,
                        'box': d_boxes[i]
                    })
                
                # Sort boxes (Column-aware)
                sorted_lines = self.sort_boxes(raw_lines, image.width if image else 1000)
                
                for line in sorted_lines:
                    full_text += line['text'] + " "
                    lines.append(line)
            
            elif isinstance(res_item, list):
                # Old structure: list of [box, (text, conf)]
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
                
                sorted_lines = self.sort_boxes(raw_lines, image.width if image else 1000)
                for line in sorted_lines:
                    full_text += line['text'] + " "
                    lines.append(line)
                
        return full_text.strip(), lines

    def sort_boxes(self, lines, image_width):
        """
        Sort text boxes in reading order.
        Handles dual-column layouts by detecting a vertical gap.
        """
        if not lines:
            return []
            
        # Helper to normalize box to points [[x,y], [x,y], ...]
        def get_points(box):
            # If flat [x1, y1, x2, y2]
            if len(box) == 4 and isinstance(box[0], (int, float, np.number)):
                x1, y1, x2, y2 = box
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            return box

        # Normalize all boxes first
        for line in lines:
            line['box'] = get_points(line['box'])
            
        # Basic Y-sort to find approximate range
        # box format is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Center Y: average of all Ys
        
        def get_center(box):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (sum(xs)/len(xs), sum(ys)/len(ys))

        # Check for columns
        # Histogram of X-coordinates?
        # Simpler: Check if we can split boxes into two clear groups by X
        # Gap threshold: e.g., 5% of width?
        
        # We only try to split if we have enough lines to justify it
        if len(lines) > 10:
            centers = [get_center(l['box']) for l in lines]
            xs = [c[0] for c in centers]
            
            # Simple 2-Means clustering or Gap detection
            # Let's try to find a vertical gap in the middle 20-80% of width
            # Sort by X
            sorted_by_x = sorted(centers, key=lambda p: p[0])
            mid_x = image_width / 2
            
            # Count items Left and Right of center
            left_items = [l for l in lines if get_center(l['box'])[0] < mid_x]
            right_items = [l for l in lines if get_center(l['box'])[0] >= mid_x]
            
            # Warning: This is a harsh split. A centered title might be split.
            # Refinement: Only split if there is an actual GAP near the middle.
            # Or reliance on Surya blocks is better?
            # Since Surya failed us on HPForcast (1 block), we MUST be smart here if we suspect columns.
            # But HPForcast is slides (single column usually).
            # MultiDomain is paper (dual column).
            
            # Heuristic: Check overlapping Y ranges between Left and Right sets?
            # If Left and Right sets have significant Y overlap, it's likely columns.
            # If Left is all above Right, it's single column.
            
            if left_items and right_items:
                y_min_l = min([min([p[1] for p in l['box']]) for l in left_items])
                y_max_l = max([max([p[1] for p in l['box']]) for l in left_items])
                y_min_r = min([min([p[1] for p in l['box']]) for l in right_items])
                y_max_r = max([max([p[1] for p in l['box']]) for l in right_items])
                
                overlap = max(0, min(y_max_l, y_max_r) - max(y_min_l, y_min_r))
                # If they overlap significantly (e.g. > 30% of height), treat as columns
                total_h = max(y_max_l, y_max_r) - min(y_min_l, y_min_r)
                
                if total_h > 0 and (overlap / total_h) > 0.3:
                    # Likely Columns -> Sort Left then Right
                    # Recursive sort on each column
                    return self.sort_boxes(left_items, mid_x) + self.sort_boxes(right_items, image_width - mid_x)

        # Standard Sort: Top-down, then Left-right
        # We group lines by Y (tolerance ~10 px)
        def sort_key(line):
             # Bin Y to merge lines
             c = get_center(line['box'])
             return (int(c[1] / 10) * 10, c[0])
             
        return sorted(lines, key=sort_key)

    def extract_text_batch(self, image: Image.Image, bboxes: List[List[float]]) -> List[Tuple[str, list]]:
        """
        Batch extract text from multiple bboxes in a single image.
        More efficient than calling extract_text() multiple times.

        Returns: List of (text, lines) tuples in same order as input bboxes
        """
        if not bboxes:
            return []

        # Crop all regions first
        crops = []
        for bbox in bboxes:
            img_w, img_h = image.size
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(img_w, int(x2)); y2 = min(img_h, int(y2))

            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(np.array(crop))
            else:
                crops.append(None)

        # Filter valid crops for batch OCR
        valid_indices = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_indices]

        # Batch OCR call - PaddleOCR accepts list of images
        if valid_crops:
            batch_results = self.ocr.ocr(valid_crops)
        else:
            batch_results = []

        # Map results back to original order
        results = [("", []) for _ in bboxes]

        for idx, result_idx in enumerate(valid_indices):
            if idx < len(batch_results) and batch_results[idx]:
                text, lines = self._parse_ocr_result(batch_results[idx], image.width)
                results[result_idx] = (text, lines)

        return results

    def _parse_ocr_result(self, result, image_width):
        """Parse single OCR result into text and lines."""
        full_text = ""
        lines = []

        if not result:
            return full_text, lines

        res_item = result[0] if isinstance(result, list) and result else result

        if isinstance(res_item, dict):
            d_texts = res_item.get('rec_texts', [])
            d_scores = res_item.get('rec_scores', [])
            d_boxes = res_item.get('rec_boxes', res_item.get('dt_polys', res_item.get('dt_boxes', [])))

            raw_lines = []
            count = min(len(d_texts), len(d_boxes)) if d_boxes else len(d_texts)
            for i in range(count):
                raw_lines.append({
                    'text': d_texts[i],
                    'confidence': d_scores[i] if i < len(d_scores) else 0.0,
                    'box': d_boxes[i] if i < len(d_boxes) else [[0,0],[0,0],[0,0],[0,0]]
                })

            sorted_lines = self.sort_boxes(raw_lines, image_width)
            for line in sorted_lines:
                full_text += line['text'] + " "
                lines.append(line)

        elif isinstance(res_item, list):
            raw_lines = []
            for line in res_item:
                coord = line[0]
                try:
                    text, conf = line[1]
                except (ValueError, IndexError):
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = 0.0
                raw_lines.append({'text': text, 'confidence': conf, 'box': coord})

            sorted_lines = self.sort_boxes(raw_lines, image_width)
            for line in sorted_lines:
                full_text += line['text'] + " "
                lines.append(line)

        return full_text.strip(), lines
