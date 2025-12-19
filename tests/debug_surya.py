from surya.layout import LayoutPredictor, FoundationPredictor
from PIL import Image
import torch

def debug_layout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    predictor = LayoutPredictor(FoundationPredictor(device=device))
    
    # Create a dummy image
    img = Image.new('RGB', (500, 500), color='white')
    
    print("Running prediction...")
    results = predictor([img])
    result = results[0]
    
    print("Result Type:", type(result))
    print("Result Dir:", dir(result))
    
    # Try to print attributes likely to contain boxes
    for attr in ['bboxes', 'boxes', 'layout', 'polygons']:
        if hasattr(result, attr):
            print(f"Found attribute '{attr}':", getattr(result, attr))

if __name__ == "__main__":
    debug_layout()
