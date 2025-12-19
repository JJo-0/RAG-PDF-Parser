---
name: markdown-gen
description: 마크다운 문서 생성 스킬. 기술 문서, 블로그 포스트, README, 보고서 작성. Claude 또는 Local LLM 활용.
allowed-tools: Read, Write, Bash
---

# Markdown Generation Skill (마크다운 생성 스킬)

고품질 마크다운 문서 자동 생성.

## 문서 유형별 템플릿

### 1. README.md
```markdown
# 프로젝트명

> 한 줄 설명

[![License](badge)](link)
[![Version](badge)](link)

## 📋 목차
- [설치](#설치)
- [사용법](#사용법)
- [기여](#기여)

## ✨ 특징
- 특징 1
- 특징 2

## 🚀 설치

```bash
pip install package-name
```

## 📖 사용법

```python
from package import Module
```

## 🤝 기여
...

## 📄 라이선스
MIT License
```

### 2. 기술 문서
```markdown
# API 문서

## 개요
...

## 인증
...

## 엔드포인트

### GET /api/resource

**요청**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| id | string | Yes | 리소스 ID |

**응답**
```json
{
  "status": "success",
  "data": {}
}
```

**예시**
```bash
curl -X GET "https://api.example.com/resource?id=123"
```
```

### 3. 블로그 포스트
```markdown
---
title: "제목"
date: 2025-01-01
tags: [tag1, tag2]
---

# 제목

## TL;DR
> 핵심 요약

## 서론
...

## 본론

### 섹션 1
...

### 섹션 2
...

## 결론
...

## 참고 자료
- [링크1](url)
```

### 4. 보고서
```markdown
# 보고서 제목

**작성자**: 이름  
**작성일**: 2025-01-01  
**버전**: 1.0

---

## 요약 (Executive Summary)
...

## 1. 배경
...

## 2. 분석
...

## 3. 결론 및 권고사항
...

## 부록
...
```

## 마크다운 스타일 가이드

### 제목 구조
```markdown
# H1 - 문서 제목 (1개만)
## H2 - 주요 섹션
### H3 - 하위 섹션
#### H4 - 세부 항목
```

### 강조
```markdown
**굵게** - 중요 키워드
*기울임* - 강조, 용어
`코드` - 인라인 코드, 명령어
~~취소선~~ - 삭제된 내용
```

### 목록
```markdown
- 순서 없는 목록
  - 중첩 항목

1. 순서 있는 목록
   1. 중첩 항목
```

### 표
```markdown
| 왼쪽 정렬 | 가운데 정렬 | 오른쪽 정렬 |
|:----------|:----------:|----------:|
| 내용 | 내용 | 내용 |
```

### 코드 블록
````markdown
```python
# 언어 지정
def example():
    pass
```
````

### 인용
```markdown
> 인용문
> 
> — 출처
```

### 체크리스트
```markdown
- [x] 완료 항목
- [ ] 미완료 항목
```

## 프롬프트 템플릿

### README 생성
```
다음 프로젝트에 대한 README.md를 작성하세요:

프로젝트명: {name}
설명: {description}
기술 스택: {tech_stack}
주요 기능: {features}

포함할 섹션:
- 프로젝트 소개
- 설치 방법
- 사용 예시
- 기여 가이드
- 라이선스
```

### 기술 문서 생성
```
다음 API/함수에 대한 기술 문서를 작성하세요:

코드:
{code}

포함할 내용:
- 함수/클래스 설명
- 파라미터 설명
- 반환값
- 예외 처리
- 사용 예시
```

### 블로그 포스트 생성
```
다음 주제로 기술 블로그 포스트를 작성하세요:

주제: {topic}
대상 독자: {audience}
길이: {length}

톤: 친근하고 교육적
구조: 서론 - 본론 - 결론
```

## CLI 사용

### README 생성
```bash
python scripts/markdown_gen.py readme \
    --name "Project Name" \
    --description "설명" \
    --output README.md
```

### 문서 변환
```bash
# 코드에서 문서 생성
python scripts/markdown_gen.py from-code src/module.py --output docs/module.md

# JSON 스키마에서 문서 생성
python scripts/markdown_gen.py from-schema schema.json --output docs/api.md
```

## 자동화

### Git Hook (커밋 시 README 업데이트)
```bash
#!/bin/bash
# .git/hooks/pre-commit

if git diff --cached --name-only | grep -q "src/"; then
    python scripts/markdown_gen.py update-readme
    git add README.md
fi
```

### CI/CD (문서 자동 빌드)
```yaml
# .github/workflows/docs.yml
- name: Generate Documentation
  run: |
    python scripts/markdown_gen.py generate-all --output docs/
```

## 관련 파일

- `scripts/markdown_gen.py`: 마크다운 생성 스크립트
- `templates/`: 문서 템플릿 모음
- `templates/readme.md`: README 템플릿
- `templates/blog.md`: 블로그 템플릿
- `templates/report.md`: 보고서 템플릿
