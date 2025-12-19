# RAG PDF Parser

## 프로젝트 개요
RAG(Retrieval-Augmented Generation)용 PDF 파서. 학술 논문의 레이아웃, 표, 이미지를 보존하며 마크다운으로 변환.

## 기술 스택
| 역할 | 라이브러리 |
|------|-----------|
| Layout Detection | Surya (`vikparuchuri/surya_layout2`) |
| OCR | PaddleOCR (한국어/영어) |
| VLM Caption | Ollama (`qwen3-vl:8b`) |
| Translation | Ollama (`gpt-oss:20b`) |
| PDF 처리 | PyMuPDF (fitz) |
| Viewer | Streamlit |

## 프로젝트 구조
```
RAG/
├── main.py                      # CLI 진입점
├── streamlit_viewer.py          # 마크다운 뷰어 (번역/중복검사 포함)
├── src/
│   ├── layout/
│   │   └── detector.py          # Surya 레이아웃 감지
│   ├── text/
│   │   └── extractor.py         # PaddleOCR 텍스트 추출 (배치 지원)
│   ├── captioning/
│   │   └── vlm.py               # AI 캡션 생성 (비동기 배치)
│   ├── translation/
│   │   └── translator.py        # 번역 모듈 (영↔한, 문단별)
│   ├── dedup/
│   │   └── deduplicator.py      # 중복 검사 (PDF/이미지/URL)
│   ├── processing/
│   │   ├── aggregator.py        # 파이프라인 오케스트레이터
│   │   └── heading.py           # 제목 레벨 감지
│   ├── table/
│   │   └── extractor.py         # 표 추출 (tabled)
│   └── chart/
│       └── extractor.py         # 차트 데이터 추출
├── output/                      # 파싱 결과물
│   ├── *.md                     # 마크다운 파일
│   ├── images/                  # 추출된 이미지
│   └── .dedup_db.json           # 중복 검사 DB
└── tests/                       # 테스트/디버그 스크립트
```

## 주요 파일
- `main.py`: PDF 파싱 파이프라인 진입점
- `src/processing/aggregator.py`: 모든 모듈 통합, 배치 OCR/VLM 최적화
- `src/layout/detector.py`: Surya 모델로 레이아웃 블록 감지
- `src/text/extractor.py`: PaddleOCR + Column-Aware 정렬
- `src/captioning/vlm.py`: Ollama VLM 비동기 캡션 생성
- `src/translation/translator.py`: 영↔한 번역, 문단별 진행, 병렬 표시
- `src/dedup/deduplicator.py`: SHA-256/Perceptual hash 기반 중복 검사
- `streamlit_viewer.py`: 결과 미리보기 + 실시간 번역 + 중복 검사 UI

## 사용법

### PDF 파싱
```bash
python main.py "path/to/document.pdf" --output_dir output
```

### 뷰어 실행
```bash
python -m streamlit run streamlit_viewer.py
```

## 주요 기능

### 1. 번역 (Translation)
- 영어 ↔ 한국어 양방향
- 문단별 번역 (진행률 표시)
- 원문 아래 번역문 병렬 표시
- 모델: `gpt-oss:20b`

### 2. 중복 검사 (Deduplication)
- PDF: SHA-256 파일 해시
- 이미지: Perceptual hash (유사 이미지 감지)
- URL: 정규화된 해시
- JSON DB 저장 (`output/.dedup_db.json`)

### 3. Streamlit 뷰어
- 📖 Viewer: 마크다운 렌더링 + 번역
- 🔍 Duplicates: 중복 검사 + DB 관리

## 작업 시 주의사항
- **Ollama 서버**: VLM/번역 기능 사용 시 Ollama가 실행 중이어야 함
  ```bash
  ollama serve
  ollama pull qwen3-vl:8b
  ollama pull gpt-oss:20b
  ```
- **PaddleOCR GPU**: CUDA 사용 시 `paddlepaddle-gpu` 설치 필요
- **배치 처리**: `aggregator.py`에서 OCR/VLM 배치 처리로 성능 최적화됨

## 데이터 플로우
```
PDF → [PyMuPDF] → 이미지 (200 DPI)
        ↓
    [Surya] 레이아웃 감지
        ↓
    ┌─────────────────────┐
    │ Text → [Batch OCR]  │
    │ Image → [Batch VLM] │
    └─────────────────────┘
        ↓
    Markdown 통합 → 출력
        ↓
    [Streamlit Viewer]
        ├── 번역 (문단별)
        └── 중복 검사
```

## 성능 최적화 포인트
1. **PaddleOCR**: `extract_text_batch()` - 여러 영역 한 번에 OCR
2. **VLM Caption**: `caption_batch()` - 비동기 병렬 처리 (max 3개)
3. **I/O 최적화**: PIL Image 직접 전달로 디스크 재읽기 제거
4. **번역 캐싱**: Streamlit session_state에 번역 결과 캐싱
