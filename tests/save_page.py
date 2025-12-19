import fitz
from PIL import Image

pdf_path = "tests/다중 도메인 비전 시스템 기반 제조 환경 안전 모니터링을 위한 동적 3D 작업자 자세 정합기법.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
pix.save("tests/page_0.png")
print("Saved tests/page_0.png")
