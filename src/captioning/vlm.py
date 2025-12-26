"""
VLM Image Captioning module for RAG PDF Parser.

Uses Ollama VLM models to generate structured captions for figures, charts, and tables.
Enhanced with type-aware prompts and hallucination control.
"""

import base64
import os
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Dict, Any
from PIL import Image
from io import BytesIO


# Structured prompt templates for different content types
STRUCTURED_PROMPTS = {
    "figure": """Analyze this figure/image carefully. Respond ONLY in valid JSON format:
{
    "type": "diagram|photo|illustration|screenshot|schematic",
    "title": "title if visible, otherwise 'untitled'",
    "description": "concise description in 50-100 words",
    "key_elements": ["element1", "element2", "element3"],
    "text_in_image": ["any visible text labels"],
    "colors_used": ["main colors if relevant"]
}

IMPORTANT: Only describe what you can clearly see. Use "unknown" for uncertain elements.""",

    "chart": """Analyze this chart/graph carefully. Respond ONLY in valid JSON format:
{
    "chart_type": "bar|line|pie|scatter|histogram|heatmap|other",
    "title": "chart title if visible",
    "x_axis": {"label": "axis label", "type": "numeric|categorical|time"},
    "y_axis": {"label": "axis label", "type": "numeric|categorical"},
    "data_summary": "key trend or pattern in 30-50 words",
    "legend_items": ["item1", "item2"],
    "notable_values": ["any specific values mentioned"]
}

IMPORTANT: Only report values you can clearly read. Use "unknown" for unclear elements.""",

    "table": """Analyze this table image carefully. Respond ONLY in valid JSON format:
{
    "title": "table title if visible",
    "columns": ["column1", "column2", "column3"],
    "row_count": 0,
    "header_row": true,
    "summary": "what this table shows in 30-50 words",
    "key_findings": ["notable data point 1", "notable data point 2"]
}

IMPORTANT: Only report structure you can clearly see. Count rows carefully.""",

    "formula": """Analyze this mathematical formula or equation. Respond ONLY in valid JSON format:
{
    "type": "equation|formula|expression",
    "latex": "LaTeX representation if possible",
    "description": "what this formula represents",
    "variables": ["var1", "var2"],
    "domain": "physics|math|chemistry|engineering|other"
}

IMPORTANT: Use standard LaTeX notation. Mark uncertain parts with '?'.""",

    "generic": """Describe this image in detail.
- If it contains a chart or graph, explain the data trends
- If it contains a diagram, explain the components and relationships
- If it contains text, transcribe the key parts
- Note any labels, legends, or annotations

Be concise but thorough. Focus on facts visible in the image."""
}


class ImageCaptioner:
    """
    VLM-based image captioner with structured output support.
    """

    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        host: str = "http://localhost:11434",
        max_concurrent: int = 3,
        timeout: int = 60,
        structured_output: bool = True
    ):
        """
        Initialize the Image Captioner.

        Args:
            model: Ollama VLM model name
            host: Ollama API host URL
            max_concurrent: Maximum concurrent VLM requests
            timeout: Request timeout in seconds
            structured_output: Use structured JSON prompts
        """
        self.model = model
        self.host = host
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.structured_output = structured_output
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

        print(f"Loading Image Captioner: {model} (concurrent={max_concurrent})")

    def _encode_image(self, image_source: Union[str, Image.Image, bytes]) -> Optional[str]:
        """Encode image to base64 from various sources."""
        try:
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    return None
                with open(image_source, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')

            elif isinstance(image_source, Image.Image):
                # Resize if too large (VLM memory optimization)
                max_dim = 1024
                if max(image_source.size) > max_dim:
                    ratio = max_dim / max(image_source.size)
                    new_size = (int(image_source.width * ratio), int(image_source.height * ratio))
                    image_source = image_source.resize(new_size, Image.Resampling.LANCZOS)

                buffer = BytesIO()
                image_source.save(buffer, format="PNG", optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

            elif isinstance(image_source, bytes):
                return base64.b64encode(image_source).decode('utf-8')

        except Exception as e:
            print(f"Error encoding image: {e}")
        return None

    def _get_prompt(self, content_type: str = "generic") -> str:
        """Get appropriate prompt for content type."""
        return STRUCTURED_PROMPTS.get(content_type, STRUCTURED_PROMPTS["generic"])

    def caption(
        self,
        image_source: Union[str, Image.Image],
        content_type: str = "generic",
        prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate a caption for the image using Ollama VLM.

        Args:
            image_source: File path or PIL Image
            content_type: Type of content ("figure", "chart", "table", "formula", "generic")
            prompt: Optional custom prompt (overrides content_type if provided)
            options: Optional Ollama generation options (temperature, num_predict, etc.)

        Returns:
            Caption string or None if failed
        """
        import requests

        image_base64 = self._encode_image(image_source)
        if not image_base64:
            return None

        try:
            # Use custom prompt if provided, otherwise use content_type prompt
            if prompt is None:
                prompt = self._get_prompt(content_type)
            else:
                prompt = prompt

            # Default options for captions
            default_options = {
                "temperature": 0.3,  # Lower for more deterministic output
                "num_predict": 500
            }

            # Override with custom options if provided
            if options:
                default_options.update(options)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": default_options
            }

            # Add format constraint if options contain 'format'
            if options and 'format' in options:
                payload["format"] = options.pop('format')  # Remove from options, add to top level

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            # Qwen3-VL may return content in 'thinking' field instead of 'response'
            caption = result.get("response", "").strip()
            if not caption:
                # Fallback to 'thinking' field (Qwen3 models)
                caption = result.get("thinking", "").strip()

            # Try to extract clean text from JSON response
            if self.structured_output and content_type != "generic":
                caption = self._format_structured_response(caption, content_type)

            return caption

        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

    def caption_structured(
        self,
        image_source: Union[str, Image.Image],
        content_type: str = "figure"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a structured caption as a dictionary.

        Args:
            image_source: File path or PIL Image
            content_type: Type of content

        Returns:
            Dictionary with structured caption data, or None if failed
        """
        import requests

        image_base64 = self._encode_image(image_source)
        if not image_base64:
            return None

        try:
            prompt = self._get_prompt(content_type)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 600
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "").strip()

            # Parse JSON from response
            return self._parse_json_response(response_text)

        except Exception as e:
            print(f"Error generating structured caption: {e}")
            return None

    def caption_with_context(
        self,
        image_source: Union[str, Image.Image],
        anchor: str,
        surrounding_text: str = "",
        content_type: str = "generic"
    ) -> Optional[str]:
        """
        Generate caption grounded in document context.

        Args:
            image_source: File path or PIL Image
            anchor: Citation anchor (e.g., "[@p3_fig1]")
            surrounding_text: Text from adjacent blocks for context
            content_type: Type of content

        Returns:
            Context-aware caption string
        """
        import requests

        image_base64 = self._encode_image(image_source)
        if not image_base64:
            return None

        try:
            context_prompt = f"""Document context:
- Reference: {anchor}
- Nearby text: "{surrounding_text[:300]}..."

Based on this context, describe what this image shows and how it relates to the surrounding text.
Be specific and reference any labels, values, or elements visible in the image."""

            payload = {
                "model": self.model,
                "prompt": context_prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except Exception as e:
            print(f"Error generating contextual caption: {e}")
            return None

    def _format_structured_response(self, response: str, content_type: str) -> str:
        """Format structured JSON response as readable text."""
        data = self._parse_json_response(response)
        if not data:
            return response  # Return raw if parsing failed

        if content_type == "chart":
            parts = []
            if data.get("chart_type"):
                parts.append(f"{data['chart_type'].title()} chart")
            if data.get("title"):
                parts.append(f"titled '{data['title']}'")
            if data.get("data_summary"):
                parts.append(f"showing {data['data_summary']}")
            return ". ".join(parts) if parts else response

        elif content_type == "table":
            parts = []
            if data.get("title"):
                parts.append(f"Table: {data['title']}")
            if data.get("row_count"):
                parts.append(f"{data['row_count']} rows")
            if data.get("columns"):
                parts.append(f"Columns: {', '.join(data['columns'][:5])}")
            if data.get("summary"):
                parts.append(data['summary'])
            return ". ".join(parts) if parts else response

        elif content_type == "figure":
            parts = []
            if data.get("type"):
                parts.append(data['type'].title())
            if data.get("description"):
                parts.append(data['description'])
            if data.get("key_elements"):
                parts.append(f"Key elements: {', '.join(data['key_elements'][:3])}")
            return ". ".join(parts) if parts else response

        return response

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from VLM response."""
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    async def _caption_async(
        self,
        session: aiohttp.ClientSession,
        image_source: Union[str, Image.Image],
        semaphore: asyncio.Semaphore,
        content_type: str = "generic"
    ) -> Optional[str]:
        """Async caption generation for a single image."""
        async with semaphore:
            image_base64 = self._encode_image(image_source)
            if not image_base64:
                return None

            prompt = self._get_prompt(content_type)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500
                }
            }

            try:
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        caption = result.get("response", "").strip()

                        if self.structured_output and content_type != "generic":
                            caption = self._format_structured_response(caption, content_type)

                        return caption
            except Exception as e:
                print(f"Async caption error: {e}")
            return None

    async def caption_batch_async(
        self,
        image_sources: List[Union[str, Image.Image]],
        content_types: Optional[List[str]] = None
    ) -> List[Optional[str]]:
        """
        Generate captions for multiple images concurrently.

        Args:
            image_sources: List of file paths or PIL Images
            content_types: Optional list of content types for each image

        Returns:
            List of captions in same order as input
        """
        if not image_sources:
            return []

        if content_types is None:
            content_types = ["generic"] * len(image_sources)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._caption_async(session, img, semaphore, ct)
                for img, ct in zip(image_sources, content_types)
            ]
            results = await asyncio.gather(*tasks)

        return results

    def caption_batch(
        self,
        image_sources: List[Union[str, Image.Image]],
        content_types: Optional[List[str]] = None
    ) -> List[Optional[str]]:
        """
        Synchronous wrapper for batch caption generation.

        Args:
            image_sources: List of file paths or PIL Images
            content_types: Optional list of content types for each image

        Returns:
            List of captions in same order as input
        """
        if not image_sources:
            return []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.caption_batch_async(image_sources, content_types)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.caption_batch_async(image_sources, content_types)
                )
        except RuntimeError:
            return asyncio.run(self.caption_batch_async(image_sources, content_types))
