from surya.layout import LayoutPredictor, FoundationPredictor
from PIL import Image
import numpy as np
import torch

class LayoutDetector:
    def __init__(self, model_path="vikparuchuri/surya_layout2"):
        """
        Initialize Surya Layout Detector.
        """
        print(f"Loading Layout Detector model: {model_path}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Using device: {device}")
        
        self.foundation = FoundationPredictor(device=device)
        self.model = LayoutPredictor(self.foundation)

    def detect(self, image: Image.Image):
        """
        Detect layout regions in the image.
        Returns a list of blocks with label, bbox, and reading order.
        """
        # Surya expects list of images or single image
        # .predict() is likely the method
        results = self.model([image]) # or .predict([image])
        
        # We only processed one image
        result = results[0]
        
        blocks = []
        # result is a LayoutResult object with 'bboxes'
        # bboxes is a list of LayoutBox objects
        
        blocks = []
        if hasattr(result, 'bboxes'):
            # Valid for newer Surya versions
            for block in result.bboxes:
                blocks.append({
                    "label": block.label,      # Text, Title, Picture, Table, etc.
                    "bbox": block.bbox,        # [x1, y1, x2, y2]
                    # order usually implied by list order or explicit field? 
                    # LayoutBox might not have 'order', but Surya sorts them usually.
                    "order": getattr(block, 'order', None)
                })
        elif hasattr(result, 'layout'):
             # Fallback
             for block in result.layout:
                 blocks.append({
                     "label": block.label,
                     "bbox": block.bbox,
                     "order": getattr(block, 'order', None)
                 })
        else:
             print(f"Warning: Unknown Surya result format. Keys: {dir(result)}")
            
        return blocks
