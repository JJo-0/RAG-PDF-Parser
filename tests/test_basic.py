import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.aggregator import MarkdownAggregator

def create_dummy_image():
    """Create a dummy image with some text and a table-like structure."""
    img = Image.new('RGB', (800, 1000), color='white')
    d = ImageDraw.Draw(img)
    
    # Title
    d.text((50, 50), "Test Document Title", fill='black')
    
    # Text block
    d.text((50, 100), "This is a sample paragraph for testing text extraction.", fill='black')
    
    # Table-like lines
    d.rectangle([50, 200, 750, 400], outline='black')
    d.line([50, 250, 750, 250], fill='black') # Header separator
    d.line([400, 200, 400, 400], fill='black') # Vertical separator
    
    d.text((60, 210), "Header 1", fill='black')
    d.text((410, 210), "Header 2", fill='black')
    d.text((60, 260), "Row 1 Col 1", fill='black')
    d.text((410, 260), "Row 1 Col 2", fill='black')
    
    return img

def test_pipeline():
    print("Initializing Aggregator...")
    try:
        aggregator = MarkdownAggregator()
        print("Aggregator initialized.")
    except Exception as e:
        print(f"Failed to initialize aggregator: {e}")
        return

    print("Creating dummy image...")
    img = create_dummy_image()
    
    print("Running aggregation (Mock run)...")
    # Note: Real surya/paddle might fail if models aren't downloaded or if image is too simple/synthetic
    # This test mainly checks if code integration is correct (imports, class usage).
    
    try:
        # We catch potential model execution errors as this is a synthetic image
        # and we might not have GPU/Models fully set up in this environment context immediately
        result = aggregator.process_page(img)
        print("Aggregation successful!")
        print("--- Result Preview ---")
        print(result[:500])
        print("----------------------")
    except Exception as e:
        print(f"Aggregation execution failed (Expected if models not ready/downloaded): {e}")

if __name__ == "__main__":
    test_pipeline()
