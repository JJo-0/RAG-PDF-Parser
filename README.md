# RAG PDF Parser v2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG(Retrieval-Augmented Generation)용 고급 PDF 파서. 학술 논문의 레이아웃, 표, 이미지를 보존하며 마크다운 및 JSONL로 변환합니다.

## 주요 특징

- 🎯 **정확한 레이아웃 감지**: Surya를 사용한 고정밀 레이아웃 분석
- 📝 **다국어 OCR**: PaddleOCR 기반 한국어/영어/중국어 지원
- 🖼️ **AI 이미지 캡션**: Ollama VLM을 활용한 구조화된 캡션 생성
- 🌐 **양방향 번역**: Ollama를 사용한 영어↔한국어 번역
- 📊 **표/차트 추출**: 표와 차트 데이터 구조화 추출
- 🔗 **Provenance Tracking**: IR(Intermediate Representation) 기반 출처 추적
- 💾 **영속적 캐싱**: SQLite 기반 OCR/VLM 결과 캐싱
- ⚡ **GPU 스케줄링**: OOM 방지를 위한 리소스 관리

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| Layout Detection | Surya (`vikparuchuri/surya_layout2`) |
| OCR | PaddleOCR (한국어/영어/중국어) |
| VLM Caption | Ollama (`qwen3-vl:8b`) |
| Translation | Ollama (`gpt-oss:20b`) |
| PDF 처리 | PyMuPDF (fitz) |
| Viewer | Streamlit |

## 설치 방법

### 1. uvx 사용 (권장)

```bash
# uvx 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 설치 및 실행
uvx rag-pdf-parser input.pdf
```

### 2. 일반 설치

#### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

#### Windows
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

### 3. Docker 사용

```bash
# CPU 버전
docker-compose up rag-parser

# GPU 버전
docker-compose --profile gpu up rag-parser-gpu
```

## 사용법

### CLI 기본 사용

```bash
# 기본 처리 (마크다운 출력)
python main.py input.pdf

# JSONL 출력 (RAG 파이프라인용)
python main.py input.pdf --output_mode jsonl

# 마크다운 + JSONL + 청킹
python main.py input.pdf --output_mode both --chunk --with_anchors

# 고품질 처리 (고해상도 + 번역)
python main.py input.pdf --dpi 300 --translate --target_lang en

# 배치 처리
for pdf in *.pdf; do
    python main.py "$pdf" --output_mode both --chunk
done
```

### CLI 옵션

```
필수 인자:
  input_path              입력 PDF 파일 경로

출력 옵션:
  --output_dir DIR        출력 디렉토리 (기본: output/)
  --output_mode MODE      출력 모드: markdown, jsonl, both (기본: markdown)
  --with_anchors          마크다운에 앵커 포함 [@p1_fig3]

청킹 옵션:
  --chunk                 IR 기반 청킹 활성화
  --chunk_size N          청크 크기 (기본: 1000)
  --chunk_overlap N       청크 오버랩 (기본: 100)

번역 옵션:
  --translate             번역 활성화
  --target_lang LANG      대상 언어: ko, en (기본: en)
  --bilingual             이중 언어 출력

중복 제거:
  --dedup                 중복 블록 제거

처리 옵션:
  --dpi N                 PDF 렌더링 DPI (기본: 200)
  --ocr_lang LANG         OCR 언어: korean, en, ch (기본: korean)
  --vlm_model MODEL       VLM 모델 (기본: qwen3-vl:8b)
```

### Streamlit 뷰어

```bash
# 뷰어 실행
streamlit run streamlit_viewer.py

# 또는
python -m streamlit run streamlit_viewer.py
```

브라우저에서 `http://localhost:8501` 접속

### Docker 사용

```bash
# Streamlit 뷰어 실행
docker-compose up rag-parser

# CLI 처리
docker-compose run --rm rag-parser python main.py /app/data/input.pdf

# GPU 버전
docker-compose --profile gpu up rag-parser-gpu
```

## 프로젝트 구조

```
RAG/
├── main.py                      # CLI 진입점
├── streamlit_viewer.py          # 웹 뷰어
├── pyproject.toml               # 프로젝트 설정 (uvx 지원)
├── Dockerfile                   # Docker 이미지
├── docker-compose.yml           # Docker 구성
├── setup.sh / setup.ps1         # 설치 스크립트
├── src/
│   ├── models/                  # IR 데이터 모델
│   │   ├── block.py             # IRBlock, IRPage, IRDocument
│   │   └── chunk.py             # IRChunk, ChunkingConfig
│   ├── layout/
│   │   └── detector.py          # Surya 레이아웃 감지
│   ├── text/
│   │   └── extractor.py         # PaddleOCR 텍스트 추출
│   ├── captioning/
│   │   └── vlm.py               # Ollama VLM 캡션
│   ├── translation/
│   │   └── translator.py        # 번역 모듈
│   ├── processing/
│   │   ├── ir_processor.py      # 메인 파이프라인
│   │   ├── chunking.py          # IR 기반 청킹
│   │   ├── scheduler.py         # GPU 스케줄러
│   │   └── heading.py           # 제목 감지
│   ├── output/
│   │   └── writer.py            # 출력 writer (MD/JSONL)
│   ├── cache/
│   │   └── persistent.py        # SQLite 캐시
│   ├── table/
│   │   └── extractor.py         # 표 추출
│   ├── chart/
│   │   └── extractor.py         # 차트 추출
│   └── dedup/
│       └── deduplicator.py      # 중복 제거
├── scripts/                     # 유틸리티 스크립트
├── templates/                   # 문서 템플릿
├── tests/                       # 테스트
│   └── test_pipeline.py         # 파이프라인 테스트
└── output/                      # 출력 결과
```

## IR (Intermediate Representation) 구조

### IRBlock
```python
{
    "doc_id": "e3115d56",           # 문서 ID (SHA256 해시)
    "page": 1,                      # 페이지 번호
    "block_id": "p1_b0",            # 블록 ID
    "type": "text",                 # 타입: text, title, table, figure, chart
    "bbox": [50, 100, 500, 200],    # 경계 박스 [x1, y1, x2, y2]
    "reading_order": 0,             # 읽기 순서
    "text": "...",                  # 추출된 텍스트
    "confidence": 0.95,             # 신뢰도
    "source_hash": "a3f2c1...",    # 콘텐츠 해시
    "anchor": "[@p1_txt0]",         # 인용 앵커
    "caption": "...",               # VLM 캡션 (이미지용)
    "ocr_lines": [...]              # OCR 라인 메타데이터
}
```

### IRChunk (RAG용)
```python
{
    "chunk_id": "e3115d56_c0",
    "doc_id": "e3115d56",
    "page_range": [1, 2],
    "block_ids": ["p1_b0", "p1_b1"],
    "section": "Introduction",
    "text": "...",
    "token_count": 256,
    "anchors": ["[@p1_txt0]", "[@p1_txt1]"]
}
```

## 출력 형식

### 1. Markdown (`.md`)
- 섹션 구조 보존
- 이미지/표/차트 임베딩
- 옵션: 앵커 포함 `[@p1_fig3]`

### 2. JSONL (`.jsonl`)
- RAG 파이프라인 ready
- 완전한 provenance 메타데이터
- 구조화된 블록 단위 출력

### 3. Chunks JSONL (`.chunks.jsonl`)
- 임베딩 ready
- 토큰 수 계산 포함
- 섹션/페이지 범위 메타데이터

### 4. Metadata JSON (`.meta.json`)
- 문서 전체 메타데이터
- 처리 통계
- 언어/페이지 정보

## Ollama 모델 설정

```bash
# Ollama 서버 실행
ollama serve

# 필수 모델 다운로드
ollama pull qwen3-vl:8b      # VLM 캡션
ollama pull gpt-oss:20b      # 번역

# 선택 모델
ollama pull qwen2.5-coder:7b # 코드 분석
ollama pull qwen3:8b         # 일반 분석
ollama pull mistral:7b       # 데이터 분석
```

## GPU 지원

### CUDA (NVIDIA GPU)

```bash
# GPU 버전 설치
pip install -e ".[gpu]"

# 또는
pip install paddlepaddle-gpu
```

### Docker GPU

```bash
# NVIDIA Container Toolkit 설치 필요
docker-compose --profile gpu up
```

## 성능 최적화

1. **배치 OCR**: `extract_text_batch()` - 여러 영역 동시 처리
2. **비동기 VLM**: `caption_batch()` - 병렬 캡션 생성 (max 3)
3. **영속적 캐싱**: SQLite 기반 OCR/VLM 결과 재사용
4. **GPU 스케줄링**: OOM 방지를 위한 단계별 리소스 관리
5. **좌표 매핑 최적화**: crop-relative → page-absolute 변환

## 테스트

```bash
# 전체 테스트 실행
python tests/test_pipeline.py

# 개별 모듈 테스트
pytest tests/

# 실제 PDF 테스트
python main.py tests/sample.pdf --output_mode both --chunk
```

## 환경 변수

```bash
# Ollama 서버 주소
OLLAMA_HOST=http://localhost:11434

# 모델 소스 체크 비활성화 (빠른 시작)
DISABLE_MODEL_SOURCE_CHECK=True

# 캐시 데이터베이스 경로
CACHE_DB_PATH=output/.cache.db
```

## 문제 해결

### Windows 인코딩 오류
```powershell
# PowerShell에서 UTF-8 설정
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding
```

### CUDA OOM 오류
```bash
# DPI 낮추기
python main.py input.pdf --dpi 150

# 또는 GPU 스케줄러 설정 조정
# src/processing/scheduler.py에서 max_concurrent 줄이기
```

### Ollama 연결 실패
```bash
# Ollama 서버 상태 확인
ollama list

# 서버 재시작
pkill ollama
ollama serve
```

## 라이선스

MIT License

## 기여

PR과 Issue는 환영합니다!

## 관련 프로젝트

- [Surya OCR](https://github.com/VikParuchuri/surya)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Ollama](https://github.com/ollama/ollama)

## 변경 이력

### v2.0.0 (2024-12-22)
- IR(Intermediate Representation) 아키텍처 도입
- JSONL 출력 형식 지원
- 영속적 캐싱 (SQLite)
- GPU 스케줄러 추가
- 구조화된 VLM 프롬프트
- 청킹 및 앵커 지원
- Docker 및 uvx 지원

### v1.0.0 (2024-12-01)
- 초기 릴리스
- 기본 PDF → Markdown 변환
- OCR 및 레이아웃 감지
