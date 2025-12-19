---
name: code-review
description: 코드 리뷰 자동화 스킬. Python/JavaScript 코드 분석, 버그 탐지, 최적화 제안. Ollama qwen2.5-coder 또는 Claude 활용.
allowed-tools: Read, Write, Bash, Grep, Glob
---

# Code Review Skill (코드 리뷰 스킬)

로컬 LLM 또는 Claude를 활용한 자동 코드 리뷰.

## 지원 모델

| 모델 | 용도 | 실행 환경 |
|------|------|----------|
| `qwen2.5-coder:7b` | 빠른 리뷰 | Local (Ollama) |
| `qwen2.5-coder:32b` | 심층 분석 | Local (Ollama) |
| `deepseek-coder-v2` | 복잡한 코드 | Local (Ollama) |
| Claude | 종합 리뷰 | API |

## 리뷰 체크리스트

### 1. 코드 품질
- [ ] 함수/클래스 명명 규칙
- [ ] 주석 및 문서화
- [ ] 코드 중복
- [ ] 복잡도 (Cyclomatic Complexity)

### 2. 버그 및 취약점
- [ ] Null/None 처리
- [ ] 예외 처리
- [ ] 타입 오류 가능성
- [ ] 보안 취약점 (SQL Injection, XSS 등)

### 3. 성능
- [ ] 알고리즘 효율성
- [ ] 메모리 사용
- [ ] I/O 최적화
- [ ] 캐싱 가능성

### 4. 유지보수성
- [ ] 모듈화
- [ ] 의존성 관리
- [ ] 테스트 가능성

## 프롬프트 템플릿

### 일반 리뷰
```
다음 코드를 리뷰하세요:

1. 버그 또는 잠재적 문제점
2. 성능 개선 가능 영역
3. 코드 스타일 및 가독성
4. 보안 고려사항

각 항목에 대해 구체적인 라인 번호와 개선 제안을 제시하세요.

코드:
```python
{code}
```
```

### 특정 관점 리뷰
```
다음 코드를 {aspect} 관점에서 분석하세요:

{code}

분석 결과:
- 문제점:
- 개선안:
- 예시 코드:
```

## 출력 형식

```markdown
## 코드 리뷰 결과

### 🔴 Critical (즉시 수정 필요)
- **Line 42**: SQL Injection 취약점
  ```python
  # Before
  query = f"SELECT * FROM users WHERE id = {user_id}"
  
  # After
  query = "SELECT * FROM users WHERE id = ?"
  cursor.execute(query, (user_id,))
  ```

### 🟡 Warning (개선 권장)
- **Line 15-20**: 중복 코드 발견
  - 함수로 추출 권장

### 🟢 Suggestion (선택적 개선)
- **Line 8**: 타입 힌트 추가 권장
  ```python
  def process(data: List[Dict]) -> Optional[str]:
  ```

### 📊 메트릭
- Cyclomatic Complexity: 12 (권장: <10)
- Lines of Code: 150
- 함수 수: 8
```

## CLI 사용

### 단일 파일
```bash
python scripts/code_review.py src/main.py --model qwen2.5-coder:7b
```

### 디렉토리 전체
```bash
python scripts/code_review.py src/ --recursive --output review_report.md
```

### Git diff 리뷰
```bash
git diff HEAD~1 | python scripts/code_review.py --stdin
```

## 자동화 통합

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ai-code-review
        name: AI Code Review
        entry: python scripts/code_review.py
        language: python
        types: [python]
```

### GitHub Actions
```yaml
- name: AI Code Review
  run: |
    python scripts/code_review.py --changed-files --output pr_review.md
```

## 언어별 설정

### Python
```python
PYTHON_RULES = {
    "style": "PEP 8",
    "type_hints": True,
    "docstrings": "Google style",
    "max_line_length": 88  # Black default
}
```

### JavaScript/TypeScript
```python
JS_RULES = {
    "style": "ESLint recommended",
    "type_checking": True,  # TypeScript
    "async_handling": True
}
```

## 관련 스크립트

- `scripts/code_review.py`: 메인 리뷰 스크립트
- `scripts/complexity_analyzer.py`: 복잡도 분석
- `scripts/security_scan.py`: 보안 취약점 스캔
