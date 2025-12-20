#!/usr/bin/env python
"""
Data Analysis Automation Script.
Analyze CSV/JSON/Excel data with automatic EDA and insights.

Usage:
    python scripts/data_analysis.py eda data.csv --output report.md
    python scripts/data_analysis.py visualize data.csv --charts all
    python scripts/data_analysis.py analyze data.csv --type correlation
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Install with: pip install pandas")


class DataAnalyzer:
    """Automated data analysis with LLM insights."""

    def __init__(self, model: str = "mistral:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048}
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load data from various formats."""
        if not PANDAS_AVAILABLE:
            return None

        ext = Path(filepath).suffix.lower()

        try:
            if ext == '.csv':
                return pd.read_csv(filepath)
            elif ext == '.json':
                return pd.read_json(filepath)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(filepath)
            elif ext == '.parquet':
                return pd.read_parquet(filepath)
            else:
                print(f"Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def basic_stats(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistics."""
        stats = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        }

        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

        # Categorical summary
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            stats["categorical"] = {}
            for col in cat_cols[:5]:  # Limit
                stats["categorical"][col] = {
                    "unique": df[col].nunique(),
                    "top": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "top_freq": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
                }

        return stats

    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> Dict:
        """Detect outliers in numeric columns."""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                outliers[col] = {
                    "count": int(outlier_mask.sum()),
                    "percentage": round(outlier_mask.mean() * 100, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2)
                }

        return outliers

    def correlation_analysis(self, df: pd.DataFrame, target: str = None) -> Dict:
        """Analyze correlations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        result = {
            "matrix": corr_matrix.round(3).to_dict()
        }

        if target and target in numeric_cols:
            target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
            result["target_correlations"] = target_corr.round(3).to_dict()

        # Find high correlations
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": round(corr_val, 3)
                    })

        result["high_correlations"] = high_corr
        return result

    def generate_insights(self, stats: Dict) -> str:
        """Generate LLM insights from statistics."""
        prompt = f"""다음 데이터 분석 결과에서 인사이트를 추출하세요:

데이터 크기: {stats['shape']['rows']} 행, {stats['shape']['columns']} 열
결측치: {json.dumps(stats['missing_pct'], indent=2)}

수치형 변수 통계:
{json.dumps(stats.get('numeric_summary', {}), indent=2)[:2000]}

다음 형식으로 답변하세요:

## 주요 발견사항
1. (데이터 품질 관련)
2. (분포/패턴 관련)
3. (주의해야 할 점)

## 추천 분석 방향
- (다음으로 수행하면 좋을 분석)

인사이트:"""

        return self._call_llm(prompt)

    def generate_eda_report(self, df: pd.DataFrame, filepath: str) -> str:
        """Generate full EDA report."""
        stats = self.basic_stats(df)
        outliers = self.detect_outliers(df)
        correlations = self.correlation_analysis(df)
        insights = self.generate_insights(stats)

        report = f"""# 📊 데이터 분석 리포트

**파일**: `{filepath}`
**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. 데이터 개요

| 항목 | 값 |
|------|-----|
| 행 수 | {stats['shape']['rows']:,} |
| 열 수 | {stats['shape']['columns']} |

### 컬럼 정보

| 컬럼 | 타입 | 결측치 | 결측률 |
|------|------|--------|--------|
"""
        for col in stats['columns'][:20]:  # Limit
            dtype = stats['dtypes'].get(col, 'unknown')
            missing = stats['missing'].get(col, 0)
            missing_pct = stats['missing_pct'].get(col, 0)
            report += f"| {col} | {dtype} | {missing:,} | {missing_pct}% |\n"

        if len(stats['columns']) > 20:
            report += f"\n*...and {len(stats['columns']) - 20} more columns*\n"

        # Numeric summary
        if 'numeric_summary' in stats:
            report += "\n## 2. 수치형 변수 통계\n\n"
            num_cols = list(stats['numeric_summary'].keys())[:10]
            report += "| 컬럼 | 평균 | 표준편차 | 최소 | 최대 |\n"
            report += "|------|------|---------|------|------|\n"
            for col in num_cols:
                s = stats['numeric_summary'][col]
                report += f"| {col} | {s.get('mean', 0):.2f} | {s.get('std', 0):.2f} | {s.get('min', 0):.2f} | {s.get('max', 0):.2f} |\n"

        # Outliers
        report += "\n## 3. 이상치 탐지 (IQR 방법)\n\n"
        outlier_cols = [c for c, v in outliers.items() if v['count'] > 0]
        if outlier_cols:
            report += "| 컬럼 | 이상치 수 | 비율 |\n"
            report += "|------|----------|------|\n"
            for col in outlier_cols[:10]:
                o = outliers[col]
                report += f"| {col} | {o['count']:,} | {o['percentage']}% |\n"
        else:
            report += "이상치가 발견되지 않았습니다.\n"

        # High correlations
        report += "\n## 4. 높은 상관관계 (|r| > 0.7)\n\n"
        if correlations['high_correlations']:
            for hc in correlations['high_correlations'][:10]:
                report += f"- **{hc['col1']}** ↔ **{hc['col2']}**: {hc['correlation']}\n"
        else:
            report += "높은 상관관계가 발견되지 않았습니다.\n"

        # LLM Insights
        report += f"""
---

## 5. 🤖 AI 인사이트

{insights}

---

*Generated by Data Analysis Tool*
"""
        return report


def cmd_eda(args):
    """Run EDA."""
    if not PANDAS_AVAILABLE:
        print("❌ pandas required. Install: pip install pandas numpy")
        return

    analyzer = DataAnalyzer(model=args.model)
    df = analyzer.load_data(args.input)

    if df is None:
        return

    print(f"📊 Analyzing: {args.input}")
    print(f"   Shape: {df.shape}")

    report = analyzer.generate_eda_report(df, args.input)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved: {args.output}")
    else:
        print(report)


def cmd_analyze(args):
    """Run specific analysis."""
    if not PANDAS_AVAILABLE:
        print("❌ pandas required")
        return

    analyzer = DataAnalyzer(model=args.model)
    df = analyzer.load_data(args.input)

    if df is None:
        return

    if args.type == "correlation":
        result = analyzer.correlation_analysis(df, args.target)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.type == "outliers":
        result = analyzer.detect_outliers(df)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.type == "stats":
        result = analyzer.basic_stats(df)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Data Analysis Tool")
    subparsers = parser.add_subparsers(dest="command")

    # eda
    p_eda = subparsers.add_parser("eda", help="Exploratory Data Analysis")
    p_eda.add_argument("input", help="Data file")
    p_eda.add_argument("-o", "--output", help="Output report file")
    p_eda.add_argument("-m", "--model", default="mistral:7b", help="LLM model")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Specific analysis")
    p_analyze.add_argument("input", help="Data file")
    p_analyze.add_argument("-t", "--type", choices=["correlation", "outliers", "stats"], default="stats")
    p_analyze.add_argument("--target", help="Target column for correlation")
    p_analyze.add_argument("-m", "--model", default="mistral:7b", help="LLM model")

    args = parser.parse_args()

    if args.command == "eda":
        cmd_eda(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
