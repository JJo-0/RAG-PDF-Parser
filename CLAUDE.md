# RAG PDF Parser

## 프로젝트 개요
RAG(Retrieval-Augmented Generation)용 PDF 파서. 학술 논문의 레이아웃, 표, 이미지를 보존하며 마크다운 및 구조화된 JSONL로 변환.

**v2.0 주요 특징:**
- IR (Intermediate Representation) 기반 파이프라인
- 블록별 provenance 추적 (page, bbox, reading_order, confidence)
- 사전 청킹된 RAG-ready 출력
- 구조화된 VLM 캡션 (JSON 스키마)
- 영속 캐시 (SQLite)

## 기술 스택
| 역할 | 라이브러리 |
|------|-----------|
| Layout Detection | Qwen3-VL Vision Language Model (JSON-forced 문서 파싱) |
| OCR | PaddleOCR (한국어/영어/중국어) |
| VLM Caption | Ollama (`qwen3-vl:8b`) - 구조화된 JSON 출력 |
| Translation | Ollama (`gpt-oss:20b`) |
| PDF 처리 | PyMuPDF (fitz) |
| Viewer | Streamlit |
| Cache | SQLite |

## 프로젝트 구조
```
RAG/
├── main.py                          # CLI 진입점 (IR 파이프라인)
├── streamlit_viewer.py              # 마크다운 뷰어
├── src/
│   ├── models/                      # IR 데이터 모델
│   │   ├── block.py                 # IRBlock, IRPage, IRDocument
│   │   └── chunk.py                 # IRChunk, ChunkingConfig
│   ├── config.py                    # ProcessorConfig
│   ├── layout/
│   │   ├── base_parser.py           # 문서 파서 추상 인터페이스
│   │   └── qwen_parser.py           # Qwen3-VL 문서 파서
│   ├── text/extractor.py            # PaddleOCR 텍스트 추출
│   ├── captioning/vlm.py            # 구조화된 VLM 캡션 생성
│   ├── translation/translator.py   # 번역 모듈
│   ├── dedup/deduplicator.py        # 중복 검사
│   ├── processing/
│   │   ├── ir_processor.py          # IR 파이프라인 프로세서 (핵심)
│   │   ├── chunking.py              # IR 인식 청킹
│   │   ├── page_merger.py           # LLM 기반 페이지 병합
│   │   ├── scheduler.py             # GPU 스테이지 스케줄러
│   │   └── heading.py               # 제목 레벨 감지
│   ├── output/
│   │   └── writer.py                # 다중 출력 형식 (MD/JSONL)
│   ├── cache/
│   │   └── persistent.py            # SQLite 영속 캐시
│   ├── table/extractor.py           # 표 추출
│   └── chart/extractor.py           # 차트 데이터 추출
├── scripts/                         # CLI 도구 모음
├── templates/                       # 문서 템플릿
└── output/                          # 파싱 결과물
```

## 사용법

### 기본 사용 (Markdown 출력)
```bash
python main.py "path/to/document.pdf"
```

### IR 기반 전체 출력 (권장)
```bash
python main.py document.pdf --output_mode both --chunk --with_anchors
```

### 옵션 설명
```bash
# 출력 모드
--output_mode {markdown,jsonl,both}  # markdown: 기존 호환, both: 전체 출력

# Provenance 추적
--with_anchors                        # [@p1_fig3] 형식 citation anchor 추가

# 청킹 (RAG 파이프라인용)
--chunk                               # 사전 청킹된 chunks.jsonl 생성
--chunk_size 1000                     # 청크 크기 (토큰)
--chunk_overlap 100                   # 청크 간 오버랩

# 번역
--translate                           # 번역 포함
--target_lang {en,ko}                 # 대상 언어

# 중복 제거
--dedup                               # 중복 문서 스킵

# 페이지 병합 (NEW!)
--merge_pages                         # 페이지 경계 끊긴 문장 자동 병합 (LLM 기반)

# 처리 옵션
--dpi 200                             # PDF 렌더링 DPI
--vlm_model qwen3-vl:8b               # VLM 모델
```

### 출력 구조
```
output/
  {doc_id}/
    document.md          # 마크다운 (anchor 포함)
    blocks.jsonl         # IRBlock per line
    chunks.jsonl         # 사전 청킹된 결과
    metadata.json        # 문서 메타데이터
    images/              # 추출된 이미지
```

### 뷰어 실행
```bash
python -m streamlit run streamlit_viewer.py
```

## IR 데이터 모델

### IRBlock (블록 단위)
```python
{
    "doc_id": "abc123def456",
    "page": 1,
    "block_id": "p1_b3",
    "type": "text|title|figure|table|chart|formula",
    "bbox": [100, 200, 500, 400],
    "reading_order": 3,
    "text": "추출된 텍스트...",
    "confidence": 0.95,
    "anchor": "[@p1_txt3]",
    "source_hash": "a1b2c3d4"
}
```

### IRChunk (청크 단위)
```python
{
    "chunk_id": "abc123_c0",
    "doc_id": "abc123def456",
    "page_range": [1, 2],
    "block_ids": ["p1_b3", "p1_b4", "p2_b0"],
    "section": "Introduction",
    "text": "청크 텍스트...",
    "token_count": 856,
    "anchors": ["[@p1_txt3]", "[@p1_txt4]"]
}
```

## 스킬 (Skills)

### 1. Translation (번역)
```bash
python scripts/translate_file.py doc.md --direction en2ko --bilingual
python scripts/batch_translate.py ./docs/ --direction en2ko --parallel
```

### 2. Code Review (코드 리뷰)
```bash
python scripts/code_review.py src/main.py
python scripts/code_review.py src/ --recursive --output review.md
```

### 3. Research (논문 분석)
```bash
python scripts/research.py analyze paper.md --output notes/
python scripts/research.py queue add paper.pdf --priority high
```

### 4. Data Analysis (데이터 분석)
```bash
python scripts/data_analysis.py eda data.csv --output report.md
```

### 5. Markdown Generation (문서 생성)
```bash
python scripts/markdown_gen.py readme --name "Project" --output README.md
python scripts/markdown_gen.py from-code src/module.py --output docs/
```

## 작업 시 주의사항

### Ollama 모델
```bash
ollama serve
ollama pull qwen3-vl:8b      # VLM 캡션 (구조화된 JSON)
ollama pull gpt-oss:20b      # 번역
ollama pull qwen2.5-coder:7b # 코드 리뷰
```

### GPU/CUDA
- PaddleOCR GPU: `paddlepaddle-gpu` 설치 필요 (OCR 성능 향상)
- GPU 스케줄러가 OOM 방지

## 데이터 플로우 (v2.0)
```
PDF → [PyMuPDF] → 이미지 (DPI 설정 가능)
        ↓
    [Qwen3-VL Document Parser] JSON-forced 문서 파싱
        ↓
    ┌──────────────────────────────┐
    │ IRBlock 생성 (provenance)    │
    ├──────────────────────────────┤
    │ Text → VLM 추출 (OCR X)     │
    │ Table → [표 구조 인식]       │
    │ Image → [Structured VLM]    │
    └──────────────────────────────┘
        ↓
    IRDocument (전체 메타데이터)
        ↓
    [OutputWriter]
        ├── document.md (with anchors)
        ├── blocks.jsonl
        └── chunks.jsonl
        ↓
    [Streamlit Viewer]
        ├── 번역 (문단별)
        └── 중복 검사
```

## 성능 최적화 포인트
1. **PaddleOCR**: `extract_text_batch()` - 배치 OCR + 좌표 매핑
2. **VLM Caption**: `caption_batch()` - 비동기 병렬 처리 + 구조화된 프롬프트
3. **GPU Scheduler**: 스테이지별 동시성 제어로 OOM 방지
4. **Persistent Cache**: SQLite 기반 OCR/VLM 결과 캐싱
5. **IR Pipeline**: 메타데이터 보존으로 재처리 비용 절감
