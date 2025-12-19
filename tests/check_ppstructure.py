import cv2
import os
from paddleocr import PaddleOCR

# The error message suggested PPStructureV3 might exist
try:
    from paddleocr import PPStructureV3 as PPStructure
    print("Imported PPStructureV3")
except ImportError:
    try:
        from paddleocr import PPStructure
        print("Imported PPStructure")
    except ImportError:
        print("Could not import PPStructure or PPStructureV3")
        PPStructure = None

        # Try initializing with correct V3 arguments
        # use_region_detection=True enables layout analysis (SER)
        table_engine = PPStructure(
            show_log=True, 
            image_orientation=True, 
            lang='korean', 
            use_region_detection=True,
            use_table_recognition=True
        )
        print("PPStructureV3 initialized successfully.")
        
        # Test on a real PDF page
        import fitz
        from PIL import Image
        import numpy as np
        
        pdf_path = "tests/다중 도메인 비전 시스템 기반 제조 환경 안전 모니터링을 위한 동적 3D 작업자 자세 정합기법.pdf"
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)
        
        print("Running inference on page 0...")
        result = table_engine(img_np)
        
        print(f"Result type: {type(result)}")
        if result:
            print(f"Number of blocks: {len(result)}")
            for i, block in enumerate(result):
                print(f"Block {i}: Type={block.get('type')}, BBox={block.get('bbox')}")
                if i >= 5: break # Print first 5
        else:
             print("No blocks detected.")
             
    except Exception as e:
        print(f"Failed to init/run PPStructure instance: {e}")

