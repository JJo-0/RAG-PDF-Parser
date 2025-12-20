#!/usr/bin/env python
"""
Markdown Document Generator.
Generate README, API docs, blog posts, and reports.

Usage:
    python scripts/markdown_gen.py readme --name "Project" --output README.md
    python scripts/markdown_gen.py from-code src/module.py --output docs/module.md
    python scripts/markdown_gen.py blog --topic "AI Trends" --output blog/ai.md
"""

import argparse
import sys
import os
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MarkdownGenerator:
    """Generate various markdown documents."""

    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 4096}
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    def generate_readme(
        self,
        name: str,
        description: str = "",
        tech_stack: List[str] = None,
        features: List[str] = None,
        install_cmd: str = "pip install package"
    ) -> str:
        """Generate README.md content."""
        prompt = f"""다음 프로젝트에 대한 README.md를 작성하세요:

프로젝트명: {name}
설명: {description}
기술 스택: {', '.join(tech_stack or ['Python'])}
주요 기능: {', '.join(features or ['Feature 1'])}
설치 명령어: {install_cmd}

다음 섹션을 포함하세요:
1. 프로젝트 소개 (배지 포함)
2. 주요 기능
3. 설치 방법
4. 사용 예시
5. 기여 가이드
6. 라이선스

마크다운 형식으로 작성하세요:"""

        return self._call_llm(prompt)

    def generate_from_code(self, code: str, filename: str) -> str:
        """Generate documentation from code."""
        # Parse code for structure
        structure = self._parse_python_code(code)

        prompt = f"""다음 Python 코드에 대한 기술 문서를 작성하세요:

파일: {filename}

코드 구조:
- 클래스: {', '.join(structure.get('classes', []))}
- 함수: {', '.join(structure.get('functions', []))}
- Import: {len(structure.get('imports', []))}개

코드:
```python
{code[:4000]}
```

다음 형식으로 문서를 작성하세요:

# {filename}

## 개요
(이 모듈의 목적)

## 클래스/함수

### ClassName / function_name
**설명**: ...
**파라미터**:
- param1 (type): 설명
**반환값**: type - 설명
**예시**:
```python
# 사용 예시
```

문서:"""

        return self._call_llm(prompt)

    def generate_blog(self, topic: str, audience: str = "개발자", length: str = "medium") -> str:
        """Generate blog post."""
        length_guide = {
            "short": "500-800자",
            "medium": "1000-1500자",
            "long": "2000-3000자"
        }

        prompt = f"""다음 주제로 기술 블로그 포스트를 작성하세요:

주제: {topic}
대상 독자: {audience}
길이: {length_guide.get(length, '1000-1500자')}

다음 형식으로 작성하세요:

---
title: "제목"
date: {datetime.now().strftime('%Y-%m-%d')}
tags: [tag1, tag2, tag3]
---

# 제목

## TL;DR
> 핵심 요약 (2-3문장)

## 서론
(왜 이 주제가 중요한지)

## 본론
### 섹션 1
...
### 섹션 2
...

## 결론
(핵심 메시지 정리)

## 참고 자료
- [링크](url)

포스트:"""

        return self._call_llm(prompt)

    def generate_api_doc(self, spec: Dict) -> str:
        """Generate API documentation."""
        prompt = f"""다음 API 스펙으로 문서를 작성하세요:

{spec}

다음 형식으로 작성하세요:

# API 문서

## 개요
...

## 인증
...

## 엔드포인트

### METHOD /path

**설명**: ...

**요청**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| param | string | Yes | 설명 |

**응답**
```json
{{
  "status": "success"
}}
```

**예시**
```bash
curl -X GET "url"
```

문서:"""

        return self._call_llm(prompt)

    def _parse_python_code(self, code: str) -> Dict:
        """Parse Python code structure."""
        structure = {
            "imports": [],
            "classes": [],
            "functions": [],
            "docstring": None
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    structure["imports"].append(f"{node.module}")
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_') or node.name == '__init__':
                        structure["functions"].append(node.name)

            # Get module docstring
            if ast.get_docstring(tree):
                structure["docstring"] = ast.get_docstring(tree)

        except SyntaxError:
            pass

        return structure


def cmd_readme(args):
    """Generate README."""
    gen = MarkdownGenerator(model=args.model)

    print(f"📝 Generating README for: {args.name}")

    readme = gen.generate_readme(
        name=args.name,
        description=args.description or "",
        tech_stack=args.tech.split(',') if args.tech else None,
        features=args.features.split(',') if args.features else None,
        install_cmd=args.install or "pip install package"
    )

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(readme)
        print(f"✅ Saved: {args.output}")
    else:
        print(readme)


def cmd_from_code(args):
    """Generate docs from code."""
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        code = f.read()

    gen = MarkdownGenerator(model=args.model)

    print(f"📝 Generating docs for: {args.input}")
    doc = gen.generate_from_code(code, os.path.basename(args.input))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f"✅ Saved: {args.output}")
    else:
        print(doc)


def cmd_blog(args):
    """Generate blog post."""
    gen = MarkdownGenerator(model=args.model)

    print(f"📝 Generating blog post: {args.topic}")
    post = gen.generate_blog(
        topic=args.topic,
        audience=args.audience,
        length=args.length
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(post)
        print(f"✅ Saved: {args.output}")
    else:
        print(post)


def main():
    parser = argparse.ArgumentParser(description="Markdown Document Generator")
    subparsers = parser.add_subparsers(dest="command")

    # readme
    p_readme = subparsers.add_parser("readme", help="Generate README.md")
    p_readme.add_argument("--name", required=True, help="Project name")
    p_readme.add_argument("--description", help="Project description")
    p_readme.add_argument("--tech", help="Tech stack (comma-separated)")
    p_readme.add_argument("--features", help="Features (comma-separated)")
    p_readme.add_argument("--install", help="Install command")
    p_readme.add_argument("-o", "--output", help="Output file")
    p_readme.add_argument("-m", "--model", default="qwen3:8b", help="LLM model")

    # from-code
    p_code = subparsers.add_parser("from-code", help="Generate docs from code")
    p_code.add_argument("input", help="Source code file")
    p_code.add_argument("-o", "--output", help="Output file")
    p_code.add_argument("-m", "--model", default="qwen3:8b", help="LLM model")

    # blog
    p_blog = subparsers.add_parser("blog", help="Generate blog post")
    p_blog.add_argument("--topic", required=True, help="Blog topic")
    p_blog.add_argument("--audience", default="개발자", help="Target audience")
    p_blog.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    p_blog.add_argument("-o", "--output", help="Output file")
    p_blog.add_argument("-m", "--model", default="qwen3:8b", help="LLM model")

    args = parser.parse_args()

    if args.command == "readme":
        cmd_readme(args)
    elif args.command == "from-code":
        cmd_from_code(args)
    elif args.command == "blog":
        cmd_blog(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
