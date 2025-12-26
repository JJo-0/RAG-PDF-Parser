"""
Page Merger - Merge split sentences across page boundaries.

Uses LLM to intelligently detect and merge sentences that are split across
PDF page boundaries in markdown output.
"""

import re
from typing import List, Tuple, Optional
import requests


class PageMerger:
    """
    Merges split sentences across page boundaries using LLM analysis.

    Detects page breaks in markdown, extracts context around boundaries,
    and uses an LLM to determine if text should be merged.
    """

    MERGE_PROMPT_TEMPLATE = """### Role
You represent a specialized text editor algorithm. Your task is to merge the text from the end of one page and the beginning of the next page in a PDF document.

### Input Data
- **Previous Page Tail:** {prev_tail}
- **Next Page Head:** {next_head}

### Instructions
1. **Analyze Context:** Read the "Previous Page Tail" and "Next Page Head".
2. **Detect Split:** Determine if the sentence at the end of the previous page continues into the next page.
   - Look for hyphenated words split across pages (e.g., "commu-" + "nication").
   - Look for incomplete sentences (e.g., ending with "the", "of", or no punctuation).
3. **Remove Artifacts:** Identify and remove any page headers, footers, or page numbers that might be inserted between the split text (e.g., "Page 10", "Conference Title").
4. **Merge:** Combine the split fragments into a single coherent paragraph.
5. **Output:** Return ONLY the merged transition paragraph. Do not summarize. Do not change words unless fixing a split hyphen.

### Constraint
- If the pages are distinct sections and clearly separated (e.g., end of a chapter), output "NO_MERGE_REQUIRED".

### Example
Input:
Tail: "...system allows the ro-"
Head: "bot to navigate autonomously."

Output:
"...system allows the robot to navigate autonomously."

Now process the given text."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
        context_chars: int = 500
    ):
        """
        Initialize Page Merger.

        Args:
            model: Ollama model for merge detection
            host: Ollama API host
            timeout: Request timeout in seconds
            context_chars: Number of characters to extract around boundary
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.context_chars = context_chars

        print(f"Initialized Page Merger: {model} (context={context_chars} chars)")

    def find_page_boundaries(self, markdown: str) -> List[Tuple[int, int]]:
        """
        Find all page boundary positions in markdown.

        Args:
            markdown: Full markdown content

        Returns:
            List of (start, end) positions for each page boundary marker
        """
        # Pattern: ---\n<!-- Page X -->
        pattern = r'\n---\n<!-- Page \d+ -->\n'
        boundaries = []

        for match in re.finditer(pattern, markdown):
            boundaries.append((match.start(), match.end()))

        return boundaries

    def extract_context(
        self,
        markdown: str,
        boundary_pos: Tuple[int, int]
    ) -> Tuple[str, str]:
        """
        Extract context around a page boundary.

        Args:
            markdown: Full markdown content
            boundary_pos: (start, end) position of boundary marker

        Returns:
            (previous_tail, next_head) text around boundary
        """
        start, end = boundary_pos

        # Extract tail from previous page
        prev_start = max(0, start - self.context_chars)
        prev_tail = markdown[prev_start:start].strip()

        # Extract head from next page
        next_end = min(len(markdown), end + self.context_chars)
        next_head = markdown[end:next_end].strip()

        return prev_tail, next_head

    def should_merge(
        self,
        prev_tail: str,
        next_head: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Use LLM to determine if pages should be merged.

        Args:
            prev_tail: Text from end of previous page
            next_head: Text from start of next page

        Returns:
            (should_merge, merged_text) tuple
        """
        try:
            prompt = self.MERGE_PROMPT_TEMPLATE.format(
                prev_tail=prev_tail,
                next_head=next_head
            )

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for deterministic output
                    "num_predict": 1000
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            output = result.get("response", "").strip()

            # Check if merge is not required
            if "NO_MERGE_REQUIRED" in output:
                return False, None

            # Otherwise, return merged text
            return True, output

        except Exception as e:
            print(f"  [WARNING] Merge detection failed: {e}")
            return False, None

    def merge_pages(self, markdown: str) -> str:
        """
        Process markdown and merge split sentences across pages.

        Args:
            markdown: Original markdown content

        Returns:
            Processed markdown with merged page boundaries
        """
        print(f"Analyzing page boundaries for merge opportunities...")

        boundaries = self.find_page_boundaries(markdown)
        print(f"  Found {len(boundaries)} page boundaries")

        if not boundaries:
            return markdown

        # Process boundaries in reverse order to preserve positions
        result = markdown
        merged_count = 0

        for idx, boundary_pos in enumerate(reversed(boundaries), 1):
            page_num = len(boundaries) - idx + 2  # Page number of next page
            print(f"  Checking boundary before Page {page_num}...", end=" ")

            # Extract context
            prev_tail, next_head = self.extract_context(result, boundary_pos)

            # Check if merge is needed
            should_merge, merged_text = self.should_merge(prev_tail, next_head)

            if should_merge and merged_text:
                print(f"[MERGE]")

                # Calculate positions for replacement
                start, end = boundary_pos

                # Find where prev_tail starts
                tail_start = result.rfind(prev_tail, 0, start)
                if tail_start == -1:
                    # Fallback: use context_chars
                    tail_start = max(0, start - self.context_chars)

                # Find where next_head ends
                head_end = result.find(next_head, end) + len(next_head)
                if head_end <= end:
                    # Fallback: use context_chars
                    head_end = min(len(result), end + self.context_chars)

                # Replace with merged text
                result = (
                    result[:tail_start] +
                    "\n\n" + merged_text + "\n\n" +
                    result[head_end:]
                )

                merged_count += 1
            else:
                print(f"[SKIP]")

        print(f"  Merged {merged_count} page boundaries")

        return result

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Process a markdown file and merge pages.

        Args:
            input_path: Path to input markdown file
            output_path: Optional output path (defaults to input_path with '_merged' suffix)

        Returns:
            Path to output file
        """
        import os

        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown = f.read()

        # Process
        merged = self.merge_pages(markdown)

        # Determine output path
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_merged{ext}"

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged)

        print(f"  Saved merged markdown to: {output_path}")

        return output_path


def merge_markdown_pages(
    markdown: str,
    model: str = "qwen2.5-coder:7b",
    host: str = "http://localhost:11434"
) -> str:
    """
    Convenience function to merge pages in markdown string.

    Args:
        markdown: Markdown content
        model: Ollama model name
        host: Ollama host URL

    Returns:
        Merged markdown content
    """
    merger = PageMerger(model=model, host=host)
    return merger.merge_pages(markdown)
