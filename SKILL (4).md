---
name: data-analysis
description: 데이터 분석 자동화 스킬. CSV/JSON/Excel 데이터 분석, 시각화, 인사이트 추출. Local LLM (Mistral/Qwen) 활용.
allowed-tools: Read, Write, Bash, Python
---

# Data Analysis Skill (데이터 분석 스킬)

로컬 LLM을 활용한 데이터 분석 자동화.

## 지원 형식

| 형식 | 확장자 | 라이브러리 |
|------|--------|-----------|
| CSV | `.csv` | pandas |
| JSON | `.json` | pandas/json |
| Excel | `.xlsx`, `.xls` | openpyxl |
| Parquet | `.parquet` | pyarrow |
| SQLite | `.db` | sqlite3 |

## 분석 워크플로우

```
데이터 로드 → EDA → 전처리 → 분석 → 시각화 → 리포트 생성
```

## 자동 EDA (탐색적 데이터 분석)

### 프롬프트 템플릿
```
다음 데이터셋 정보를 분석하세요:

컬럼 정보:
{columns_info}

기초 통계:
{describe_output}

샘플 데이터:
{sample_rows}

분석 항목:
1. 데이터 개요 (행/열 수, 결측치)
2. 각 컬럼의 특성 및 분포
3. 잠재적 이상치
4. 컬럼 간 관계 추정
5. 추천 분석 방향
```

### 자동 생성 리포트
```markdown
# 📊 데이터 분석 리포트

## 1. 데이터 개요
- **행 수**: 10,000
- **열 수**: 15
- **결측치**: 3개 컬럼에서 발견

## 2. 컬럼별 분석

### 수치형 변수
| 컬럼 | 평균 | 표준편차 | 최소 | 최대 | 결측치 |
|------|------|---------|------|------|--------|
| age | 35.2 | 12.4 | 18 | 85 | 0% |
| income | 52,000 | 25,000 | 15,000 | 200,000 | 2% |

### 범주형 변수
| 컬럼 | 고유값 수 | 최빈값 | 빈도 |
|------|----------|--------|------|
| gender | 2 | M | 52% |
| region | 5 | Seoul | 35% |

## 3. 인사이트
...

## 4. 권장 분석
...
```

## 시각화 자동 생성

### 지원 차트
```python
CHART_TYPES = {
    "distribution": ["histogram", "boxplot", "violin"],
    "relationship": ["scatter", "heatmap", "pairplot"],
    "comparison": ["bar", "grouped_bar", "stacked_bar"],
    "trend": ["line", "area"],
    "composition": ["pie", "treemap"]
}
```

### 자동 차트 추천
```
데이터 특성 분석 결과:
- 수치형 변수 2개 → scatter plot 추천
- 범주형 + 수치형 → boxplot 추천
- 시계열 데이터 → line chart 추천
- 비율 데이터 → pie/donut chart 추천
```

## 분석 템플릿

### 기술 통계
```python
def generate_summary(df):
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe().to_dict()
    }
    return summary
```

### 상관관계 분석
```python
def correlation_analysis(df, target=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    if target:
        target_corr = corr_matrix[target].sort_values(ascending=False)
        return target_corr
    return corr_matrix
```

### 이상치 탐지
```python
def detect_outliers(df, columns, method="iqr"):
    outliers = {}
    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower) | (df[col] > upper)]
    return outliers
```

## CLI 사용

### 빠른 EDA
```bash
python scripts/data_analysis.py eda data.csv --output report.md
```

### 시각화 생성
```bash
python scripts/data_analysis.py visualize data.csv --charts all --output charts/
```

### 특정 분석
```bash
python scripts/data_analysis.py analyze data.csv \
    --type correlation \
    --target sales \
    --output correlation_report.md
```

## LLM 활용 분석

### 인사이트 추출
```python
def generate_insights(summary_stats, model="mistral:7b"):
    prompt = f"""
    다음 데이터 분석 결과에서 비즈니스 인사이트를 추출하세요:
    
    {summary_stats}
    
    다음 형식으로 답변하세요:
    1. 주요 발견사항 (3가지)
    2. 잠재적 문제점
    3. 권장 액션
    """
    return ollama_generate(prompt, model)
```

### SQL 쿼리 생성
```python
def generate_sql_query(question, schema):
    prompt = f"""
    테이블 스키마:
    {schema}
    
    질문: {question}
    
    이 질문에 답하는 SQL 쿼리를 작성하세요.
    """
    return ollama_generate(prompt, "qwen2.5-coder:7b")
```

## 모델 별 권장 용도

| 모델 | 용도 | 특징 |
|------|------|------|
| `mistral:7b` | 일반 분석 인사이트 | 빠름, 범용적 |
| `qwen2.5-coder:7b` | 코드/쿼리 생성 | 코딩 특화 |
| `llama3.2:8b` | 복잡한 추론 | 추론 능력 |
| Claude | 종합 리포트 | 긴 컨텍스트 |

## 관련 파일

- `scripts/data_analysis.py`: 메인 분석 스크립트
- `scripts/visualizer.py`: 시각화 생성
- `scripts/sql_generator.py`: SQL 쿼리 생성
- `templates/analysis_report.md`: 리포트 템플릿
