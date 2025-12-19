---
name: pdf-parsing
description: PDF 파싱 최적화 스킬. 학술 논문, 슬라이드 등 복잡한 문서의 레이아웃/표/이미지 추출. Surya + PaddleOCR 활용.
allowed-tools: Read, Write, Bash, Grep
---

# PDF Parsing Skill (PDF 파싱 스킬)

학술 논문 및 복잡한 문서를 위한 고품질 PDF 파싱 가이드.

## 파이프라인 개요

```
PDF → 이미지 변환 → 레이아웃 감지 → OCR → 캡션 생성 → 마크다운 조립
```

## 레이아웃 감지 규칙

### Surya 모델 사용
```python
from surya.model.detection import segformer_detector
from surya.layout import batch_layout_detection

# 지원 블록 타입
BLOCK_TYPES = ["Text", "Title", "Table", "Figure", "List", "Caption"]
```

### 다단 컬럼 처리
1. 수직 간격 분석으로 컬럼 구분
2. 좌→우, 상→하 순서로 정렬
3. 컬럼 간 텍스트 혼합 방지

## OCR 최적화

### PaddleOCR 설정
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='korean',        # 'en', 'korean', 'ch'
    use_gpu=True,
    det_db_thresh=0.3,    # 텍스트 감지 임계값
    rec_batch_num=6       # 배치 크기
)
```

### 언어별 설정
| 언어 | lang 파라미터 | 참고 |
|------|--------------|------|
| 한국어 | `korean` | 한글 + 영어 혼합 가능 |
| 영어 | `en` | 영어 전용 |
| 중국어 | `ch` | 간체 중국어 |

## Fallback 메커니즘

레이아웃 감지 실패 시:
1. **Full-Page OCR**: 전체 페이지 OCR 실행
2. **Native Image Extraction**: PyMuPDF로 원본 이미지 추출

```python
def fallback_extraction(page):
    # 1. Full-page OCR
    ocr_result = ocr.ocr(page_image)
    
    # 2. Native image extraction
    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        base_image = pdf.extract_image(xref)
```

## 표 처리

### 감지된 표 영역
1. 이미지로 크롭
2. VLM으로 표 내용 설명 생성
3. 마크다운 표 또는 이미지로 삽입

### 출력 예시
```markdown
![Table 1](images/table_1.png)
*AI Caption: 이 표는 다양한 모델의 BLEU 점수를 비교합니다...*
```

## 이미지 처리

### 추출 우선순위
1. **Native Extraction** (PyMuPDF): 원본 품질 유지
2. **Crop from Page**: 레이아웃 기반 크롭

### VLM 캡션 프롬프트
```
이 이미지를 분석하고 다음을 포함하여 설명하세요:
1. 이미지 유형 (차트/다이어그램/사진 등)
2. 주요 내용 및 데이터
3. 문서 맥락에서의 의미
```

## 마크다운 출력 구조

```markdown
# 문서 제목

## 섹션 1

텍스트 내용...

![Figure 1](images/figure_1.png)
*AI Caption: 그림 설명...*

| 열1 | 열2 |
|-----|-----|
| 데이터 | 데이터 |

## 섹션 2
...
```

## 성능 최적화

### GPU 메모리 관리
```python
import torch

# 모델 사용 후 메모리 해제
torch.cuda.empty_cache()
```

### 배치 처리
- 페이지 단위로 처리하여 메모리 관리
- 대용량 PDF는 청크로 분할

## 트러블슈팅

### 레이아웃 감지 부정확
- DPI 높이기: `dpi=300` 이상
- 전처리: 이진화, 노이즈 제거

### OCR 품질 저하
- 이미지 해상도 확인
- 언어 설정 확인
- 영역 확대하여 재처리

### 표 인식 실패
- 표 영역 수동 지정 옵션 사용
- VLM 프롬프트 상세화

## 관련 파일

- `src/layout/detector.py`: 레이아웃 감지
- `src/text/extractor.py`: OCR 추출
- `src/captioning/vlm.py`: VLM 캡션
- `main.py`: 파이프라인 통합
