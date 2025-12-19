import base64
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union
from PIL import Image
from io import BytesIO

class ImageCaptioner:
    def __init__(self, model="qwen3-vl:8b", host="http://localhost:11434", max_concurrent=3):
        self.model = model
        self.host = host
        self.max_concurrent = max_concurrent  # Limit concurrent VLM requests
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        print(f"Loading Image Captioner with model: {model} (max_concurrent={max_concurrent})")

    def _encode_image(self, image_source: Union[str, Image.Image, bytes]) -> Optional[str]:
        """Encode image to base64 from various sources."""
        try:
            if isinstance(image_source, str):
                # File path
                if not os.path.exists(image_source):
                    return None
                with open(image_source, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            elif isinstance(image_source, Image.Image):
                # PIL Image - avoid disk I/O
                buffer = BytesIO()
                image_source.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            elif isinstance(image_source, bytes):
                return base64.b64encode(image_source).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
        return None

    def caption(self, image_source: Union[str, Image.Image]) -> Optional[str]:
        """
        Generate a caption for the image using Ollama VLM.
        Accepts file path or PIL Image directly.
        """
        import requests

        image_base64 = self._encode_image(image_source)
        if not image_base64:
            return None

        try:
            prompt = "Describe this image in detail. If it is a chart or graph, explain the trends and data. If it is a diagram, explain the components."

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }

            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

    async def _caption_async(self, session: aiohttp.ClientSession, image_source: Union[str, Image.Image], semaphore: asyncio.Semaphore) -> Optional[str]:
        """Async caption generation for a single image."""
        async with semaphore:
            image_base64 = self._encode_image(image_source)
            if not image_base64:
                return None

            prompt = "Describe this image in detail. If it is a chart or graph, explain the trends and data. If it is a diagram, explain the components."

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }

            try:
                async with session.post(f"{self.host}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
            except Exception as e:
                print(f"Async caption error: {e}")
            return None

    async def caption_batch_async(self, image_sources: List[Union[str, Image.Image]]) -> List[Optional[str]]:
        """
        Generate captions for multiple images concurrently.
        Returns list of captions in same order as input.
        """
        if not image_sources:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._caption_async(session, img, semaphore)
                for img in image_sources
            ]
            results = await asyncio.gather(*tasks)

        return results

    def caption_batch(self, image_sources: List[Union[str, Image.Image]]) -> List[Optional[str]]:
        """
        Synchronous wrapper for batch caption generation.
        Use this from non-async code.
        """
        if not image_sources:
            return []

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.caption_batch_async(image_sources))
                    return future.result()
            else:
                return loop.run_until_complete(self.caption_batch_async(image_sources))
        except RuntimeError:
            # No event loop exists
            return asyncio.run(self.caption_batch_async(image_sources))
