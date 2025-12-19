# Local RAG PDF Parser (로컬 RAG PDF 파서)

RAG (Retrieval-Augmented Generation) 애플리케이션을 위해 설계된, 로컬 우선(Local-First) PDF 파싱 파이프라인입니다. 학술 논문이나 슬라이드 덱과 같은 복잡한 문서의 구조, 표(Table), 그림(Figure)을 보존하며 내용을 추출하는 데 특화되어 있습니다.

## 주요 기능 (Key Features)

1.  **Layout-Aware Extraction (레이아웃 인식 추출)**:
    -   `Surya` (또는 `PaddleOCR Structure`) 딥러닝 모델을 사용하여 텍스트, 제목, 표, 그림 등의 레이아웃 블록을 감지합니다.
2.  **Column-Aware Text Sorting (다단 컬럼 인식 정렬)**:
    -   다단(Multi-column)으로 구성된 학술 논문에서 텍스트가 뒤섞이는 것을 방지하기 위해, 수직 간격을 분석하여 읽기 순서(Reading Order)를 지능적으로 정렬합니다.
3.  **Hybrid Fallback Mechanism (하이브리드 예외 처리)**:
    -   레이아웃 감지가 실패할 경우(예: 복잡한 논문), 자동으로 **Full-Page OCR**과 **Native Image Extraction**(PyMuPDF)을 실행하여 데이터 유실을 방지합니다.
4.  **Table & Image Handling (표 및 이미지 처리)**:
    -   표와 이미지를 감지하고 크롭(Crop)하여 저장합니다.
    -   **VLM (Vision Language Model)**: `Qwen3-VL:8b` (Ollama)을 사용하여 추출된 차트나 로봇/장비 사진에 대해 상세한 **AI 캡션(AI Caption)**을 생성합니다.
5.  **Markdown Output**: 임베딩(Embedding) 및 인덱싱에 최적화된, 구조가 살아있는 Markdown 문서를 생성합니다.

## 워크플로우 개요 (Workflow Overview)

파이프라인(`main.py`)은 PDF 파일을 다음 단계로 처리합니다:

### 1. Visualization & Pre-processing
-   PDF를 시각적 분석을 위해 고해상도 이미지로 변환합니다.

### 2. Layout Detection (레이아웃 감지, `src/layout/detector.py`)
-   **Engine**: `vikparuchuri/surya_layout2` (추후 `PP-Structure`로 고도화 예정)
-   페이지를 스캔하여 `Text`, `Title`, `Table`, `Figure` 등의 좌표(Bounding Box)를 찾습니다.
-   *Note*: 레이아웃을 찾지 못하면 **Hybrid Fallback**이 트리거됩니다.

### 3. Text Extraction (텍스트 추출, `src/text/extractor.py`)
-   **Engine**: `PaddleOCR` (한국어/영어 지원)
-   감지된 블록 내의 텍스트를 추출합니다.
-   **Column-Aware Logic**: 페이지가 2단/3단 구성인지 판단하고, 사람의 읽기 흐름(좌상단 -> 좌하단 -> 우상단 -> 우하단)에 맞춰 텍스트를 재배열합니다.

### 4. Image & Table Extraction
-   **Native Fallback**: `fitz` (PyMuPDF)를 사용하여 PDF 내부의 원본 이미지 스트림을 직접 추출함으로써, 스크린샷 캡쳐보다 선명한 화질의 다이어그램을 확보합니다.

### 5. Semantic Captioning (`src/captioning/vlm.py`)
-   **Engine**: Local LLM/VLM via Ollama (기본값: `qwen3-vl:8b`).
-   추출된 모든 이미지/표에 대해 "이 차트는 무엇을 설명하는가?"에 대한 AI 설명을 생성합니다.
-   Markdown 결과물에 `*AI Caption: ...*` 형태로 추가되어, 검색 시 이미지 내용도 검색되도록 합니다.

### 6. Aggregation
-   헤딩(Heading), 텍스트, 이미지 링크, 캡션을 하나의 완성된 Markdown 문서로 통합합니다.

## 설치 방법 (Installation)

1.  **Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    # 주요 라이브러리: surya-ocr, paddlepaddle, paddleocr, pymupdf, pillow, requests
    ```

2.  **Ollama Models**:
    Ollama가 설치되어 있어야 하며, VLM 모델을 Pull 해야 합니다:
    ```bash
    ollama pull qwen3-vl:8b
    ```

## 사용법 (Usage)

터미널에서 다음 명령어로 PDF를 파싱합니다:

```bash
python main.py "path/to/document.pdf" --output_dir output
```

**결과물 구조 (Output Structure)**:
```
output/
├── document_name.md        # 최종 파싱된 마크다운 파일
└── images/                 # 추출된 그림 및 표 이미지
    ├── Figure_1_....png
    └── native_figure_....jpg
```

## 설정 (Configuration)

-   **Model Selection**: `src/captioning/vlm.py`에서 Ollama 모델을 변경할 수 있습니다.
-   **OCR Language**: `src/text/extractor.py` 내부의 `lang` 파라미터를 수정하여 한국어(`korean`) 또는 영어(`en`)로 설정할 수 있습니다.
