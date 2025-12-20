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
│   ├── layout/detector.py       # Surya 레이아웃 감지
│   ├── text/extractor.py        # PaddleOCR 텍스트 추출 (배치 지원)
│   ├── captioning/vlm.py        # AI 캡션 생성 (비동기 배치)
│   ├── translation/translator.py # 번역 모듈 (영↔한, 문단별)
│   ├── dedup/deduplicator.py    # 중복 검사 (PDF/이미지/URL)
│   ├── processing/
│   │   ├── aggregator.py        # 파이프라인 오케스트레이터
│   │   └── heading.py           # 제목 레벨 감지
│   ├── table/extractor.py       # 표 추출
│   └── chart/extractor.py       # 차트 데이터 추출
├── scripts/                     # CLI 도구 모음
│   ├── translate_file.py        # 단일 파일 번역
│   ├── batch_translate.py       # 배치 번역
│   ├── code_review.py           # AI 코드 리뷰
│   ├── research.py              # 논문 분석 도구
│   ├── data_analysis.py         # 데이터 분석
│   └── markdown_gen.py          # 마크다운 생성
├── templates/                   # 문서 템플릿
│   ├── paper_summary.md         # 논문 요약 템플릿
│   ├── blog.md                  # 블로그 템플릿
│   └── analysis_report.md       # 분석 리포트 템플릿
├── SKILL*.md                    # 스킬 정의 파일들
└── output/                      # 파싱 결과물
```

## 사용법

### PDF 파싱
```bash
python main.py "path/to/document.pdf" --output_dir output
```

### 뷰어 실행
```bash
python -m streamlit run streamlit_viewer.py
```

## 스킬 (Skills)

### 1. Translation (번역)
```bash
# 단일 파일
python scripts/translate_file.py doc.md --direction en2ko --bilingual

# 배치 처리
python scripts/batch_translate.py ./docs/ --direction en2ko --parallel
```

### 2. Code Review (코드 리뷰)
```bash
python scripts/code_review.py src/main.py
python scripts/code_review.py src/ --recursive --output review.md
```

### 3. Research (논문 분석)
```bash
# 논문 분석
python scripts/research.py analyze paper.md --output notes/

# 읽기 큐 관리
python scripts/research.py queue add paper.pdf --priority high
python scripts/research.py queue list
```

### 4. Data Analysis (데이터 분석)
```bash
python scripts/data_analysis.py eda data.csv --output report.md
python scripts/data_analysis.py analyze data.csv --type correlation
```

### 5. Markdown Generation (문서 생성)
```bash
# README 생성
python scripts/markdown_gen.py readme --name "Project" --output README.md

# 코드 문서화
python scripts/markdown_gen.py from-code src/module.py --output docs/

# 블로그 포스트
python scripts/markdown_gen.py blog --topic "AI Trends" --output blog/ai.md
```

## 작업 시 주의사항

### Ollama 모델
```bash
ollama serve
ollama pull qwen3-vl:8b      # VLM 캡션
ollama pull gpt-oss:20b      # 번역
ollama pull qwen2.5-coder:7b # 코드 리뷰
ollama pull qwen3:8b         # 일반 분석
ollama pull mistral:7b       # 데이터 분석
```

### GPU/CUDA
- PaddleOCR GPU: `paddlepaddle-gpu` 설치 필요
- Surya: CUDA 자동 감지

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
