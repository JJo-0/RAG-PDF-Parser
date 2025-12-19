import re

class HeadingDetector:
    def __init__(self):
        # Heuristic rules can be refined
        pass

    def estimate_level(self, text, bbox, font_size=None, is_bold=False):
        """
        Estimate heading level (1, 2, 3) or return None.
        Simple heuristic based on text length and capitalization for now
        since we don't always get font info from Surya/PaddleOCR easily without complex matching.
        """
        text = text.strip()
        if not text:
            return None

        # Rule 1: Short, all caps, or Title Case likely heading
        # If we had font size, we would use it primary.
        
        # Placeholder heuristic for "Text-only" input
        if len(text.split()) < 10:
             # Check for Chapter patterns
             if re.match(r'^(Chapter|Section) \d+', text, re.IGNORECASE):
                 return 1
             
             # All caps often Level 1 or 2
             if text.isupper() and len(text) > 4:
                 return 2
             
             # Title case
             if text.istitle():
                 return 3
                 
        return None
        
    def detect(self, block):
        """
        Process a block from LayoutDetector/Aggregator
        """
        label = block.get('label')
        text = block.get('text', '')
        
        if label == 'Title':
            return 1
        elif label == 'Section-header':
            return 2
        
        return self.estimate_level(text, block.get('bbox'))
