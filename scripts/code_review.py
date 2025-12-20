#!/usr/bin/env python
"""
AI Code Review Script.
Analyzes Python/JavaScript code for bugs, performance, and style issues.

Usage:
    python scripts/code_review.py src/main.py
    python scripts/code_review.py src/ --recursive --output review_report.md
    git diff HEAD~1 | python scripts/code_review.py --stdin
"""

import argparse
import sys
import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Optional
import requests
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CodeReviewer:
    """AI-powered code reviewer using Ollama."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 4096
                }
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error calling LLM: {e}"

    def analyze_complexity(self, code: str) -> Dict:
        """Analyze code complexity metrics."""
        metrics = {
            "lines": len(code.split('\n')),
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "complexity_warnings": []
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    metrics["functions"] += 1
                    # Check function length
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:
                        metrics["complexity_warnings"].append(
                            f"Function '{node.name}' is {func_lines} lines (>50)"
                        )
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics["imports"] += 1

        except SyntaxError:
            metrics["syntax_error"] = True

        return metrics

    def check_security(self, code: str) -> List[Dict]:
        """Check for common security issues."""
        issues = []

        security_patterns = [
            (r'eval\s*\(', "Use of eval() - potential code injection", "critical"),
            (r'exec\s*\(', "Use of exec() - potential code injection", "critical"),
            (r'os\.system\s*\(', "Use of os.system() - prefer subprocess", "warning"),
            (r'shell\s*=\s*True', "shell=True in subprocess - potential injection", "warning"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected", "critical"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected", "critical"),
            (r'SELECT\s+.*\s+FROM\s+.*\s*%', "Potential SQL injection (string formatting)", "critical"),
            (r'f["\'].*SELECT.*{', "Potential SQL injection (f-string)", "critical"),
            (r'\.format\(.*\).*SELECT', "Potential SQL injection (.format)", "critical"),
            (r'pickle\.loads?\s*\(', "Unsafe pickle usage", "warning"),
            (r'yaml\.load\s*\([^,]+\)', "Unsafe yaml.load (use safe_load)", "warning"),
        ]

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "line": i,
                        "message": message,
                        "severity": severity,
                        "code": line.strip()
                    })

        return issues

    def review_code(self, code: str, filename: str = "code.py") -> str:
        """Perform comprehensive code review."""
        # Get metrics
        metrics = self.analyze_complexity(code)
        security_issues = self.check_security(code)

        # Prepare prompt for LLM
        prompt = f"""다음 코드를 리뷰하세요:

파일: {filename}
라인 수: {metrics['lines']}
함수 수: {metrics['functions']}
클래스 수: {metrics['classes']}

코드:
```
{code[:8000]}  # Limit code length
```

다음 항목을 분석하세요:

1. **버그 또는 잠재적 문제점**
   - Null/None 처리 미흡
   - 예외 처리 누락
   - 타입 오류 가능성

2. **성능 개선 가능 영역**
   - 비효율적인 알고리즘
   - 불필요한 반복
   - 메모리 최적화

3. **코드 스타일 및 가독성**
   - 명명 규칙
   - 코드 중복
   - 주석 및 문서화

4. **보안 고려사항**
   - 입력 검증
   - 인증/인가
   - 데이터 노출

각 항목에 대해 구체적인 라인 번호와 개선 제안을 제시하세요.
심각도를 🔴 Critical, 🟡 Warning, 🟢 Suggestion으로 표시하세요.
"""

        llm_review = self._call_llm(prompt)

        # Build final report
        report = f"""# 🔍 코드 리뷰 결과

**파일**: `{filename}`
**분석 일시**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 📊 코드 메트릭

| 항목 | 값 |
|------|-----|
| Lines of Code | {metrics['lines']} |
| Functions | {metrics['functions']} |
| Classes | {metrics['classes']} |
| Imports | {metrics['imports']} |

"""

        # Add complexity warnings
        if metrics.get("complexity_warnings"):
            report += "### ⚠️ 복잡도 경고\n"
            for warning in metrics["complexity_warnings"]:
                report += f"- {warning}\n"
            report += "\n"

        # Add security issues
        if security_issues:
            report += "## 🔒 보안 이슈\n\n"
            for issue in security_issues:
                icon = "🔴" if issue["severity"] == "critical" else "🟡"
                report += f"{icon} **Line {issue['line']}**: {issue['message']}\n"
                report += f"```\n{issue['code']}\n```\n\n"

        # Add LLM review
        report += f"""---

## 🤖 AI 분석 결과

{llm_review}

---

*Review generated by AI Code Reviewer (model: {self.model})*
"""

        return report


def review_file(filepath: str, reviewer: CodeReviewer) -> str:
    """Review a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return reviewer.review_code(code, os.path.basename(filepath))


def review_directory(
    directory: str,
    reviewer: CodeReviewer,
    recursive: bool = True,
    extensions: List[str] = None
) -> str:
    """Review all files in a directory."""
    if extensions is None:
        extensions = ['.py', '.js', '.ts']

    path = Path(directory)
    files = []

    if recursive:
        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(path.glob(f"*{ext}"))

    # Filter out common directories
    files = [f for f in files if not any(
        part.startswith('.') or part in ['node_modules', '__pycache__', 'venv', 'env']
        for part in f.parts
    )]

    if not files:
        return "No files found to review."

    report = f"# 📁 Directory Code Review\n\n"
    report += f"**Directory**: `{directory}`  \n"
    report += f"**Files**: {len(files)}  \n\n"
    report += "---\n\n"

    for i, filepath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Reviewing {filepath.name}...")
        file_report = review_file(str(filepath), reviewer)
        report += file_report
        report += "\n\n---\n\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="AI Code Review Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/code_review.py src/main.py
    python scripts/code_review.py src/ --recursive
    python scripts/code_review.py src/main.py --model qwen2.5-coder:32b
    git diff HEAD~1 | python scripts/code_review.py --stdin
        """
    )

    parser.add_argument("path", nargs="?", help="File or directory to review")
    parser.add_argument("--stdin", action="store_true", help="Read code from stdin")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursive directory scan")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-m", "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model (default: qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "-e", "--extensions",
        nargs="+",
        default=[".py", ".js", ".ts"],
        help="File extensions (default: .py .js .ts)"
    )

    args = parser.parse_args()

    reviewer = CodeReviewer(model=args.model)

    # Process input
    if args.stdin:
        code = sys.stdin.read()
        report = reviewer.review_code(code, "stdin")
    elif args.path:
        path = Path(args.path)
        if path.is_file():
            print(f"📄 Reviewing file: {args.path}")
            report = review_file(args.path, reviewer)
        elif path.is_dir():
            print(f"📁 Reviewing directory: {args.path}")
            report = review_directory(args.path, reviewer, args.recursive, args.extensions)
        else:
            print(f"❌ Path not found: {args.path}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Report saved: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
