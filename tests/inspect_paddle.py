import paddleocr
print(f"PaddleOCR File: {paddleocr.__file__}")
print(f"PaddleOCR Dir: {dir(paddleocr)}")
try:
    import paddleocr.ppstructure
    print(f"ppstructure module: {paddleocr.ppstructure}")
    print(f"ppstructure Dir: {dir(paddleocr.ppstructure)}")
except ImportError as e:
    print(f"Could not import ppstructure: {e}")
