#!/usr/bin/env python
"""
Single file translation script.
Translates markdown/text files between English and Korean.

Usage:
    python scripts/translate_file.py input.md --direction en2ko --output output_translated.md
    python scripts/translate_file.py input.md --direction ko2en
    python scripts/translate_file.py input.md --bilingual  # Show both original and translation
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translation import Translator


def translate_file(
    input_path: str,
    output_path: str = None,
    direction: str = "auto",
    bilingual: bool = False,
    model: str = "gpt-oss:20b"
):
    """
    Translate a single file.

    Args:
        input_path: Path to input file
        output_path: Path to output file (default: input_translated.md)
        direction: 'en2ko', 'ko2en', or 'auto'
        bilingual: If True, show original + translation
        model: Ollama model name
    """
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"📄 Input: {input_path}")
    print(f"📊 Size: {len(content)} characters")

    # Initialize translator
    translator = Translator(model=model)

    # Determine source/target languages
    if direction == "auto":
        source_lang = translator.detect_language(content)
        target_lang = "en" if source_lang == "ko" else "ko"
        print(f"🔍 Auto-detected: {source_lang} → {target_lang}")
    elif direction == "en2ko":
        source_lang, target_lang = "en", "ko"
    elif direction == "ko2en":
        source_lang, target_lang = "ko", "en"
    else:
        raise ValueError(f"Invalid direction: {direction}")

    print(f"🌐 Translating: {source_lang} → {target_lang}")

    # Translate
    if bilingual:
        print("📝 Mode: Bilingual (original + translation)")

        def progress_callback(current, total):
            print(f"  Translating paragraph {current}/{total}...", end='\r')

        result = translator.translate_markdown_bilingual(
            content,
            source_lang=source_lang,
            target_lang=target_lang,
            paragraph_callback=progress_callback
        )
        print()  # New line after progress
    else:
        print("📝 Mode: Translation only")
        # Parse paragraphs and translate
        paragraphs = translator._parse_markdown_paragraphs(content)

        result_parts = []
        total = sum(1 for ptype, _ in paragraphs if ptype == 'text')
        current = 0

        for ptype, para_content in paragraphs:
            if ptype == 'special':
                result_parts.append(para_content)
            else:
                if para_content.strip():
                    current += 1
                    print(f"  Translating paragraph {current}/{total}...", end='\r')
                    translated = translator.translate(para_content, source_lang, target_lang)
                    result_parts.append(translated if translated else para_content)
                else:
                    result_parts.append(para_content)

        print()
        result = '\n'.join(result_parts)

    # Determine output path
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_translated{ext}"

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"✅ Output saved: {output_path}")
    print(f"📊 Output size: {len(result)} characters")


def main():
    parser = argparse.ArgumentParser(
        description="Translate a markdown/text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/translate_file.py paper.md --direction en2ko
    python scripts/translate_file.py doc.md --bilingual --output doc_bilingual.md
    python scripts/translate_file.py readme.md --direction auto
        """
    )

    parser.add_argument("input", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-d", "--direction",
        choices=["en2ko", "ko2en", "auto"],
        default="auto",
        help="Translation direction (default: auto)"
    )
    parser.add_argument(
        "-b", "--bilingual",
        action="store_true",
        help="Output bilingual format (original + translation)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        sys.exit(1)

    try:
        translate_file(
            input_path=args.input,
            output_path=args.output,
            direction=args.direction,
            bilingual=args.bilingual,
            model=args.model
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
