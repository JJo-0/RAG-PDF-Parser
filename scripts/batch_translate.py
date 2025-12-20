#!/usr/bin/env python
"""
Batch translation script for multiple files.
Translates all markdown/text files in a directory.

Usage:
    python scripts/batch_translate.py ./docs/ --direction en2ko --parallel
    python scripts/batch_translate.py ./papers/ --direction en2ko --output ./translated/
"""

import argparse
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translation import Translator


def translate_single_file(
    input_path: str,
    output_path: str,
    translator: Translator,
    source_lang: str,
    target_lang: str,
    bilingual: bool
) -> dict:
    """Translate a single file and return result info."""
    start_time = time.time()

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if bilingual:
            result = translator.translate_markdown_bilingual(
                content,
                source_lang=source_lang,
                target_lang=target_lang
            )
        else:
            paragraphs = translator._parse_markdown_paragraphs(content)
            result_parts = []

            for ptype, para_content in paragraphs:
                if ptype == 'special':
                    result_parts.append(para_content)
                else:
                    if para_content.strip():
                        translated = translator.translate(para_content, source_lang, target_lang)
                        result_parts.append(translated if translated else para_content)
                    else:
                        result_parts.append(para_content)

            result = '\n'.join(result_parts)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)

        elapsed = time.time() - start_time

        return {
            "status": "success",
            "input": input_path,
            "output": output_path,
            "input_size": len(content),
            "output_size": len(result),
            "elapsed": elapsed
        }

    except Exception as e:
        return {
            "status": "error",
            "input": input_path,
            "output": output_path,
            "error": str(e),
            "elapsed": time.time() - start_time
        }


def batch_translate(
    input_dir: str,
    output_dir: str = None,
    direction: str = "en2ko",
    bilingual: bool = False,
    parallel: bool = False,
    max_workers: int = 3,
    extensions: list = None,
    model: str = "gpt-oss:20b"
):
    """
    Translate all files in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory (default: input_dir_translated)
        direction: 'en2ko' or 'ko2en'
        bilingual: Output bilingual format
        parallel: Use parallel processing
        max_workers: Max parallel workers
        extensions: File extensions to process
        model: Ollama model name
    """
    if extensions is None:
        extensions = ['.md', '.txt']

    # Setup paths
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    if output_dir is None:
        output_dir = str(input_path) + "_translated"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all files
    files = []
    for ext in extensions:
        files.extend(input_path.rglob(f"*{ext}"))

    if not files:
        print(f"⚠️ No files found with extensions: {extensions}")
        return

    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files found: {len(files)}")
    print(f"🌐 Direction: {direction}")
    print(f"📝 Mode: {'Bilingual' if bilingual else 'Translation only'}")
    print(f"⚙️ Parallel: {parallel} (workers: {max_workers})")
    print()

    # Determine languages
    if direction == "en2ko":
        source_lang, target_lang = "en", "ko"
    else:
        source_lang, target_lang = "ko", "en"

    # Initialize translator
    translator = Translator(model=model, max_concurrent=max_workers)

    # Prepare file pairs
    file_pairs = []
    for f in files:
        rel_path = f.relative_to(input_path)
        out_file = output_path / rel_path
        file_pairs.append((str(f), str(out_file)))

    # Process files
    results = []
    start_time = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    translate_single_file,
                    inp, out, translator, source_lang, target_lang, bilingual
                ): inp
                for inp, out in file_pairs
            }

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                status = "✅" if result["status"] == "success" else "❌"
                print(f"[{i}/{len(files)}] {status} {Path(result['input']).name}")
    else:
        for i, (inp, out) in enumerate(file_pairs, 1):
            print(f"[{i}/{len(files)}] Processing {Path(inp).name}...")
            result = translate_single_file(
                inp, out, translator, source_lang, target_lang, bilingual
            )
            results.append(result)

            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {status} Done")

    # Summary
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    print()
    print("=" * 50)
    print("📊 Summary")
    print("=" * 50)
    print(f"  Total files: {len(files)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  ⏱️ Total time: {total_time:.1f}s")
    print(f"  📁 Output: {output_dir}")

    if error_count > 0:
        print()
        print("❌ Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['input']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch translate files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/batch_translate.py ./docs/ --direction en2ko
    python scripts/batch_translate.py ./papers/ --direction en2ko --parallel
    python scripts/batch_translate.py ./docs/ --bilingual --output ./translated/
        """
    )

    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "-d", "--direction",
        choices=["en2ko", "ko2en"],
        default="en2ko",
        help="Translation direction (default: en2ko)"
    )
    parser.add_argument(
        "-b", "--bilingual",
        action="store_true",
        help="Output bilingual format"
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Use parallel processing"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="Max parallel workers (default: 3)"
    )
    parser.add_argument(
        "-e", "--extensions",
        nargs="+",
        default=[".md", ".txt"],
        help="File extensions to process (default: .md .txt)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )

    args = parser.parse_args()

    try:
        batch_translate(
            input_dir=args.input_dir,
            output_dir=args.output,
            direction=args.direction,
            bilingual=args.bilingual,
            parallel=args.parallel,
            max_workers=args.workers,
            extensions=args.extensions,
            model=args.model
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
