"""
Translator module for RAG PDF Parser.
Provides bilingual translation using local Ollama LLM.
"""

import requests
import asyncio
import aiohttp
from typing import List, Optional, Tuple
import re


class Translator:
    def __init__(self, model="gpt-oss:20b", host="http://localhost:11434", max_concurrent=3):
        """
        Initialize Translator with Ollama LLM.

        Args:
            model: Ollama model name (default: gpt-oss:20b)
            host: Ollama server URL
            max_concurrent: Max concurrent translation requests
        """
        self.model = model
        self.host = host
        self.max_concurrent = max_concurrent
        print(f"Loading Translator with model: {model} (max_concurrent={max_concurrent})")

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character ranges.
        Returns 'ko' for Korean, 'en' for English.
        """
        korean_count = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))
        total_alpha = len(re.findall(r'[a-zA-Z\uac00-\ud7af]', text))

        if total_alpha == 0:
            return 'en'

        korean_ratio = korean_count / total_alpha
        return 'ko' if korean_ratio > 0.3 else 'en'

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build translation prompt."""
        lang_names = {'ko': 'Korean', 'en': 'English'}
        source = lang_names.get(source_lang, source_lang)
        target = lang_names.get(target_lang, target_lang)

        return f"""Translate the following text from {source} to {target}.
Only output the translation, nothing else. Preserve formatting and line breaks.

Text to translate:
{text}

Translation:"""

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> Optional[str]:
        """
        Translate text between Korean and English.

        Args:
            text: Text to translate
            source_lang: Source language ('ko' or 'en'). Auto-detected if None.
            target_lang: Target language. Opposite of source if None.

        Returns:
            Translated text or None on error.
        """
        if not text or not text.strip():
            return ""

        # Auto-detect source language
        if source_lang is None:
            source_lang = self.detect_language(text)

        # Default target is opposite of source
        if target_lang is None:
            target_lang = 'en' if source_lang == 'ko' else 'ko'

        # Skip if same language
        if source_lang == target_lang:
            return text

        prompt = self._build_prompt(text, source_lang, target_lang)

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            translation = result.get("response", "").strip()
            return translation

        except Exception as e:
            print(f"Translation error: {e}")
            return None

    async def _translate_async(
        self,
        session: aiohttp.ClientSession,
        text: str,
        source_lang: str,
        target_lang: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[str]:
        """Async translation for a single text."""
        async with semaphore:
            if not text or not text.strip():
                return ""

            prompt = self._build_prompt(text, source_lang, target_lang)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048
                }
            }

            try:
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
            except Exception as e:
                print(f"Async translation error: {e}")
            return None

    async def translate_batch_async(
        self,
        texts: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> List[Optional[str]]:
        """
        Translate multiple texts concurrently.

        Args:
            texts: List of texts to translate
            source_lang: Source language (auto-detected per text if None)
            target_lang: Target language

        Returns:
            List of translated texts in same order.
        """
        if not texts:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for text in texts:
                src = source_lang if source_lang else self.detect_language(text)
                tgt = target_lang if target_lang else ('en' if src == 'ko' else 'ko')
                tasks.append(self._translate_async(session, text, src, tgt, semaphore))

            results = await asyncio.gather(*tasks)

        return results

    def translate_batch(
        self,
        texts: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> List[Optional[str]]:
        """
        Synchronous wrapper for batch translation.
        """
        if not texts:
            return []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.translate_batch_async(texts, source_lang, target_lang)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.translate_batch_async(texts, source_lang, target_lang)
                )
        except RuntimeError:
            return asyncio.run(self.translate_batch_async(texts, source_lang, target_lang))

    def translate_markdown_bilingual(
        self,
        markdown_content: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        paragraph_callback=None
    ) -> str:
        """
        Translate markdown content and return bilingual format.
        Each paragraph shows original followed by translation.

        Args:
            markdown_content: Original markdown text
            source_lang: Source language (auto-detected if None)
            target_lang: Target language
            paragraph_callback: Optional callback(current, total) for progress

        Returns:
            Bilingual markdown with original and translated paragraphs.
        """
        if not markdown_content:
            return ""

        # Parse into paragraphs
        paragraphs = self._parse_markdown_paragraphs(markdown_content)

        # Collect texts to translate
        texts_to_translate = []
        text_indices = []

        for i, (ptype, content) in enumerate(paragraphs):
            if ptype == 'text' and content.strip():
                texts_to_translate.append(content)
                text_indices.append(i)

        # Translate paragraph by paragraph with progress
        translations = []
        total = len(texts_to_translate)

        for idx, text in enumerate(texts_to_translate):
            if paragraph_callback:
                paragraph_callback(idx + 1, total)

            translation = self.translate(text, source_lang, target_lang)
            translations.append(translation)

        # Build bilingual output
        return self._build_bilingual_output(paragraphs, translations, text_indices)

    def _parse_markdown_paragraphs(self, content: str) -> List[Tuple[str, str]]:
        """Parse markdown into paragraphs and special elements."""
        lines = content.split('\n')
        paragraphs = []
        current_para = []

        for line in lines:
            is_special = (
                line.startswith('#') or
                line.startswith('![') or
                line.startswith('<!--') or
                line.startswith('*AI ') or
                line.startswith('|') or
                line.strip() == '' or
                line.startswith('```') or
                line.startswith('>')
            )

            if is_special:
                if current_para:
                    paragraphs.append(('text', '\n'.join(current_para)))
                    current_para = []
                paragraphs.append(('special', line))
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append(('text', '\n'.join(current_para)))

        return paragraphs

    def _build_bilingual_output(
        self,
        paragraphs: List[Tuple[str, str]],
        translations: List[Optional[str]],
        text_indices: List[int]
    ) -> str:
        """Build bilingual markdown output."""
        result_lines = []
        trans_idx = 0

        for i, (ptype, content) in enumerate(paragraphs):
            if ptype == 'special':
                result_lines.append(content)
            else:
                if content.strip():
                    result_lines.append(content)
                    result_lines.append('')

                    if trans_idx < len(translations) and translations[trans_idx]:
                        result_lines.append(f"> *{translations[trans_idx]}*")
                    else:
                        result_lines.append("> *(translation unavailable)*")

                    result_lines.append('')
                    trans_idx += 1
                else:
                    result_lines.append(content)

        return '\n'.join(result_lines)

    def translate_paragraph_by_paragraph(
        self,
        paragraphs: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ):
        """
        Generator that yields (original, translation) for each paragraph.
        Useful for streaming/progress display.
        """
        for para in paragraphs:
            if para.strip():
                translation = self.translate(para, source_lang, target_lang)
                yield para, translation
            else:
                yield para, ""
