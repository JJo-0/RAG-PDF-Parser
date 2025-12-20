#!/usr/bin/env python
"""
Research & Paper Analysis CLI Tool.
Analyze academic papers, generate summaries, and manage reading queue.

Usage:
    python scripts/research.py analyze paper.pdf --output notes/
    python scripts/research.py summarize paper.md --quick
    python scripts/research.py queue add paper.pdf --priority high
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PaperAnalyzer:
    """Analyze academic papers using LLM."""

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
                "options": {"temperature": 0.3, "num_predict": 4096}
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    def quick_summary(self, text: str) -> str:
        """Generate 3-sentence summary."""
        prompt = f"""다음 논문을 3문장으로 요약하세요:
1. 연구 목적
2. 핵심 방법
3. 주요 결과

논문:
{text[:6000]}

요약:"""
        return self._call_llm(prompt)

    def deep_analysis(self, text: str) -> str:
        """Generate comprehensive analysis."""
        prompt = f"""다음 논문을 분석하세요:

{text[:8000]}

다음 형식으로 분석 결과를 작성하세요:

## 연구 배경 및 동기
(이 연구가 왜 필요한지)

## 핵심 기여 (3가지)
1.
2.
3.

## 제안 방법의 작동 원리
(핵심 알고리즘/접근법 설명)

## 실험 설계
(데이터셋, 베이스라인, 메트릭)

## 주요 결과
(정량적 결과 요약)

## 한계점 및 개선 방향

## 이 분야에서의 의의

분석:"""
        return self._call_llm(prompt)

    def extract_key_info(self, text: str) -> Dict:
        """Extract structured key information."""
        prompt = f"""다음 논문에서 정보를 추출하세요:

{text[:4000]}

JSON 형식으로 응답하세요:
{{
    "title": "논문 제목",
    "authors": ["저자1", "저자2"],
    "year": 2024,
    "venue": "학회/저널명",
    "keywords": ["키워드1", "키워드2"],
    "problem": "해결하려는 문제 (1문장)",
    "method": "제안 방법 (1문장)",
    "result": "주요 결과 (1문장)"
}}

JSON:"""
        result = self._call_llm(prompt)
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw": result}

    def compare_papers(self, paper_a: str, paper_b: str) -> str:
        """Compare two papers."""
        prompt = f"""다음 두 논문을 비교 분석하세요:

### 논문 A:
{paper_a[:3000]}

### 논문 B:
{paper_b[:3000]}

비교 항목:
1. 연구 목적
2. 방법론
3. 실험 설계
4. 결과
5. 각 논문의 장단점

비교 분석:"""
        return self._call_llm(prompt)


class ReadingQueue:
    """Manage paper reading queue."""

    def __init__(self, db_path: str = "output/.reading_queue.json"):
        self.db_path = db_path
        self.queue = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.queue, f, ensure_ascii=False, indent=2)

    def add(self, path: str, priority: str = "normal", tags: List[str] = None):
        """Add paper to queue."""
        entry = {
            "id": len(self.queue) + 1,
            "path": path,
            "filename": os.path.basename(path),
            "priority": priority,
            "tags": tags or [],
            "added": datetime.now().isoformat(),
            "status": "pending"
        }
        self.queue.append(entry)
        self._save()
        return entry

    def list(self, status: str = None) -> List[Dict]:
        """List queue entries."""
        if status:
            return [e for e in self.queue if e.get("status") == status]
        return self.queue

    def update_status(self, paper_id: int, status: str):
        """Update paper status."""
        for entry in self.queue:
            if entry["id"] == paper_id:
                entry["status"] = status
                entry["updated"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def remove(self, paper_id: int):
        """Remove paper from queue."""
        self.queue = [e for e in self.queue if e["id"] != paper_id]
        self._save()


def cmd_analyze(args):
    """Analyze a paper."""
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    print(f"📄 Analyzing: {args.input}")

    # Read content
    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()

    analyzer = PaperAnalyzer(model=args.model)

    # Generate analysis
    print("🔍 Generating analysis...")
    analysis = analyzer.deep_analysis(content)

    # Extract key info
    print("📋 Extracting key information...")
    key_info = analyzer.extract_key_info(content)

    # Build report
    report = f"""# 📄 논문 분석: {os.path.basename(args.input)}

**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**모델**: {args.model}

---

## TL;DR
"""
    if isinstance(key_info, dict) and "problem" in key_info:
        report += f"> {key_info.get('problem', '')} → {key_info.get('method', '')} → {key_info.get('result', '')}\n"

    report += f"""
## 📑 메타 정보

| 항목 | 내용 |
|------|------|
| 제목 | {key_info.get('title', 'N/A')} |
| 저자 | {', '.join(key_info.get('authors', ['N/A']))} |
| 연도 | {key_info.get('year', 'N/A')} |
| 학회/저널 | {key_info.get('venue', 'N/A')} |

### 🔑 키워드
{' '.join(['`' + kw + '`' for kw in key_info.get('keywords', [])])}

---

{analysis}

---

*Generated by Research Tool*
"""

    # Output
    if args.output:
        output_path = args.output
        if os.path.isdir(args.output):
            base = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{base}_analysis.md")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Saved: {output_path}")
    else:
        print(report)


def cmd_summarize(args):
    """Quick summarize a paper."""
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()

    analyzer = PaperAnalyzer(model=args.model)

    print(f"📄 Summarizing: {args.input}\n")
    summary = analyzer.quick_summary(content)
    print(summary)


def cmd_queue(args):
    """Manage reading queue."""
    queue = ReadingQueue()

    if args.action == "add":
        entry = queue.add(args.path, args.priority, args.tags)
        print(f"✅ Added: #{entry['id']} {entry['filename']} [{entry['priority']}]")

    elif args.action == "list":
        entries = queue.list(args.status)
        if not entries:
            print("📭 Queue is empty")
            return

        print(f"📚 Reading Queue ({len(entries)} items)\n")
        for e in entries:
            priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(e["priority"], "⚪")
            status_icon = {"pending": "⏳", "reading": "📖", "done": "✅"}.get(e["status"], "❓")
            print(f"  #{e['id']} {status_icon} {priority_icon} {e['filename']}")
            if e.get("tags"):
                print(f"       Tags: {', '.join(e['tags'])}")

    elif args.action == "done":
        if queue.update_status(args.id, "done"):
            print(f"✅ Marked as done: #{args.id}")
        else:
            print(f"❌ Not found: #{args.id}")

    elif args.action == "remove":
        queue.remove(args.id)
        print(f"🗑️ Removed: #{args.id}")


def main():
    parser = argparse.ArgumentParser(description="Research & Paper Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Deep analyze a paper")
    p_analyze.add_argument("input", help="Paper file (md/txt)")
    p_analyze.add_argument("-o", "--output", help="Output file/directory")
    p_analyze.add_argument("-m", "--model", default="qwen3:8b", help="LLM model")

    # summarize
    p_summarize = subparsers.add_parser("summarize", help="Quick summary")
    p_summarize.add_argument("input", help="Paper file")
    p_summarize.add_argument("-m", "--model", default="qwen3:8b", help="LLM model")

    # queue
    p_queue = subparsers.add_parser("queue", help="Manage reading queue")
    queue_sub = p_queue.add_subparsers(dest="action")

    q_add = queue_sub.add_parser("add", help="Add to queue")
    q_add.add_argument("path", help="Paper path")
    q_add.add_argument("-p", "--priority", choices=["high", "normal", "low"], default="normal")
    q_add.add_argument("-t", "--tags", nargs="+", help="Tags")

    q_list = queue_sub.add_parser("list", help="List queue")
    q_list.add_argument("-s", "--status", choices=["pending", "reading", "done"])

    q_done = queue_sub.add_parser("done", help="Mark as done")
    q_done.add_argument("id", type=int, help="Paper ID")

    q_remove = queue_sub.add_parser("remove", help="Remove from queue")
    q_remove.add_argument("id", type=int, help="Paper ID")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "queue":
        cmd_queue(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
