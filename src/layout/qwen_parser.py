"""
Qwen3-VL Document Parser.

Uses Qwen3-VL vision model to parse entire document pages into structured markdown,
then converts markdown to IRBlocks for the RAG pipeline.

Strategy:
1. Send full page image to Qwen3-VL with document parsing prompt
2. Receive structured markdown output
3. Parse markdown into semantic blocks (headings, tables, figures, text)
4. Assign reading order based on markdown structure
"""

import re
from typing import List, Tuple, Dict, Optional
from PIL import Image

from src.layout.base_parser import BaseDocumentParser
from src.models.block import IRBlock
from src.captioning.vlm import ImageCaptioner


class QwenVLDocumentParser(BaseDocumentParser):
    """
    Full-page document parser using Qwen3-VL vision language model.

    Uses the `qwenvl markdown` trigger prompt to convert document pages
    into structured markdown, then parses into IRBlocks.
    """

    # Document parsing prompt with structured JSON output
    DOCUMENT_PROMPT = """You are a document parser. Extract all text elements from this page into JSON format.

EXAMPLE OUTPUT (this is what you should produce):
{
  "page_elements": [
    {"type": "title", "content": "Introduction to Robotics", "metadata": {}},
    {"type": "paragraph", "content": "Authors: John Smith, Jane Doe", "metadata": {}},
    {"type": "section", "content": "1. Background", "metadata": {"section_number": "1"}},
    {"type": "paragraph", "content": "Robotics has evolved...", "metadata": {}},
    {"type": "figure", "content": "Robot diagram", "metadata": {"caption": "Fig 1: Basic robot"}}
  ]
}

ELEMENT TYPES:
- title: Page/document title
- section: Section heading (## Level 2)
- subsection: Subsection heading (### Level 3)
- paragraph: Regular text (summarize if >150 words)
- figure: Image/diagram (use caption if visible)
- table: Table (preserve structure in markdown)

INSTRUCTIONS:
1. Extract ALL visible text from the page
2. Keep paragraph content concise (first 2-3 sentences for long text)
3. Fix obvious OCR errors
4. Preserve [reference numbers]
5. Output ONLY the JSON object - no other text

NOW PARSE THIS PAGE:"""

    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        ollama_host: str = "http://localhost:11434",
        timeout: int = 120  # Longer timeout for full-page parsing
    ):
        """
        Initialize Qwen3-VL document parser.

        Args:
            model: Ollama model name (default: qwen3-vl:8b)
            ollama_host: Ollama API endpoint
            timeout: Request timeout in seconds
        """
        self.model = model
        self.captioner = ImageCaptioner(
            model=model,
            host=ollama_host,
            max_concurrent=1,  # Process one page at a time
            timeout=timeout,
            structured_output=False  # Use plain text markdown output
        )

    def parse_page(
        self,
        page_image: Image.Image,
        page_num: int,
        doc_id: str
    ) -> List[IRBlock]:
        """
        Parse full page into IRBlocks using Qwen3-VL.

        Args:
            page_image: PIL Image of the page
            page_num: 1-indexed page number
            doc_id: Document ID

        Returns:
            List of IRBlocks in reading order
        """
        # 1. Call Qwen3-VL for structured JSON
        json_str = self._call_vlm(page_image)

        # 2. Parse JSON into semantic blocks
        blocks = self._json_to_blocks(json_str, page_num, doc_id)

        # 3. Assign reading order
        for idx, block in enumerate(blocks):
            if block.reading_order is None or block.reading_order == 0:
                block.reading_order = idx

        return blocks

    def _call_vlm(self, page_image: Image.Image) -> str:
        """
        Call Qwen3-VL with document parsing prompt.

        Args:
            page_image: PIL Image

        Returns:
            JSON string with page structure
        """
        try:
            print(f"    → Calling Qwen3-VL for structured parsing...")
            # Use the existing ImageCaptioner infrastructure with custom prompt
            # Allow thinking + structured JSON output
            response = self.captioner.caption(
                image_source=page_image,
                prompt=self.DOCUMENT_PROMPT,
                options={
                    "format": "json",  # FORCE JSON output (Ollama feature)
                    "temperature": 0.2,  # Slightly higher for better generation
                    "num_predict": 8000,  # Enough for full page
                    "repeat_penalty": 1.15,  # Moderate penalty
                    "repeat_last_n": 256  # Look back far
                }
            )

            if response:
                # Extract JSON from thinking/response
                json_str = self._extract_json_from_response(response)
                if json_str:
                    print(f"    [OK] Qwen3-VL returned {len(response)} chars → extracted JSON ({len(json_str)} chars)")
                    return json_str
                else:
                    print(f"    [WARN] Could not extract valid JSON from response")
                    return ""
            else:
                print(f"    [WARN] Qwen3-VL returned empty response")
                return ""
        except Exception as e:
            print(f"    [ERROR] Qwen3-VL parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _extract_json_from_response(self, text: str) -> str:
        """
        Extract JSON object from Qwen3-VL response (may include thinking).

        The model outputs thinking first, then JSON. We need to extract
        the valid JSON object.
        """
        if not text:
            return ""

        import json
        import re

        # Strategy 1: Find JSON object with regex
        # Look for { ... } pattern that looks like our schema
        json_pattern = r'\{[\s\S]*"page_elements"[\s\S]*\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                # Try to parse as JSON
                parsed = json.loads(match)
                if "page_elements" in parsed:
                    return match
            except json.JSONDecodeError:
                continue

        # Strategy 2: Find any valid JSON object
        # Look for balanced braces
        brace_count = 0
        json_start = -1

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    # Found complete JSON object
                    candidate = text[json_start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return candidate
                    except json.JSONDecodeError:
                        json_start = -1

        print(f"    [DEBUG] Could not extract JSON from response")
        return ""

    def _json_to_blocks(
        self,
        json_str: str,
        page_num: int,
        doc_id: str
    ) -> List[IRBlock]:
        """
        Parse JSON structure into IRBlocks.

        Args:
            json_str: JSON string from Qwen3-VL
            page_num: Page number
            doc_id: Document ID

        Returns:
            List of IRBlocks
        """
        if not json_str or not json_str.strip():
            return []

        import json

        try:
            data = json.loads(json_str)
            page_elements = data.get("page_elements", [])

            if not page_elements:
                print(f"    [WARN] No page_elements found in JSON")
                return []

            blocks = []
            for idx, element in enumerate(page_elements):
                elem_type = element.get("type", "paragraph")
                content = element.get("content", "")
                level = element.get("level", 0)
                metadata = element.get("metadata", {})

                # Map element type to IRBlock type
                block_type = self._map_element_to_block_type(elem_type)

                # Generate markdown based on type
                markdown = self._format_element_markdown(elem_type, content, level, metadata)

                # Create IRBlock
                block = IRBlock(
                    doc_id=doc_id,
                    page=page_num,
                    block_id=f"p{page_num}_b{idx}",
                    type=block_type,
                    bbox=None,  # Qwen3-VL doesn't provide bbox
                    reading_order=idx,
                    text=content,
                    markdown=markdown,
                    parser_source='qwenvl',
                    raw_data=metadata if metadata else None
                )

                blocks.append(block)

            print(f"    [OK] Parsed {len(blocks)} elements from JSON")
            return blocks

        except json.JSONDecodeError as e:
            print(f"    [ERROR] JSON parse error: {e}")
            return []

    def _map_element_to_block_type(self, elem_type: str) -> str:
        """Map JSON element type to IRBlock type."""
        mapping = {
            'title': 'title',
            'author': 'text',  # Authors as text block
            'affiliation': 'text',
            'abstract': 'text',
            'section': 'title',
            'subsection': 'title',
            'paragraph': 'text',
            'table': 'table',
            'figure': 'figure'
        }
        return mapping.get(elem_type, 'text')

    def _format_element_markdown(self, elem_type: str, content: str, level: int, metadata: dict) -> str:
        """Format element as markdown based on type."""
        if elem_type == 'title':
            return f"# {content}"
        elif elem_type == 'author':
            return f"**Authors**: {content}"
        elif elem_type == 'affiliation':
            return f"*{content}*"
        elif elem_type == 'abstract':
            return f"## Abstract\n\n{content}"
        elif elem_type == 'section':
            section_num = metadata.get('section_number', '')
            if section_num:
                return f"## {section_num} {content}"
            return f"## {content}"
        elif elem_type == 'subsection':
            section_num = metadata.get('section_number', '')
            if section_num:
                return f"### {section_num} {content}"
            return f"### {content}"
        elif elem_type == 'table':
            return content  # Table already in markdown format
        elif elem_type == 'figure':
            caption = metadata.get('caption', content)
            fig_num = metadata.get('figure_number', '')
            return f"![{fig_num}: {caption}](placeholder)"
        else:  # paragraph
            return content

    def _markdown_to_blocks(
        self,
        markdown: str,
        page_num: int,
        doc_id: str
    ) -> List[IRBlock]:
        """
        Parse markdown into IRBlocks using regex patterns.

        Block type detection:
        - ## Heading → type=title
        - ### Subheading → type=title (with metadata)
        - |...|...| → type=table
        - ![...](...) → type=figure
        - $$...$$ → type=formula
        - Regular paragraphs → type=text

        Args:
            markdown: Markdown text from Qwen3-VL
            page_num: Page number
            doc_id: Document ID

        Returns:
            List of IRBlocks
        """
        if not markdown or not markdown.strip():
            return []

        blocks = []
        lines = markdown.split('\n')
        block_counter = 0

        # Patterns
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        figure_pattern = re.compile(r'^!\[([^\]]*)\]\(([^)]*)\)$')
        table_row_pattern = re.compile(r'^\|(.+)\|$')
        formula_block_pattern = re.compile(r'^\$\$(.+)\$\$$', re.DOTALL)

        # State tracking
        in_table = False
        table_lines = []
        current_paragraph = []

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                # Blank line: flush current paragraph
                if current_paragraph:
                    blocks.append(self._create_text_block(
                        current_paragraph, page_num, doc_id, block_counter
                    ))
                    block_counter += 1
                    current_paragraph = []
                continue

            # 1. Heading detection
            heading_match = heading_pattern.match(line_stripped)
            if heading_match:
                # Flush paragraph if any
                if current_paragraph:
                    blocks.append(self._create_text_block(
                        current_paragraph, page_num, doc_id, block_counter
                    ))
                    block_counter += 1
                    current_paragraph = []

                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                blocks.append(IRBlock(
                    doc_id=doc_id,
                    page=page_num,
                    block_id=f"p{page_num}_b{block_counter}",
                    type='title',
                    bbox=None,  # Qwen3-VL doesn't provide bbox
                    reading_order=block_counter,
                    text=text,
                    markdown=f"{'#' * level} {text}",
                    parser_source='qwenvl',
                    raw_data={'heading_level': level}
                ))
                block_counter += 1
                continue

            # 2. Figure detection
            figure_match = figure_pattern.match(line_stripped)
            if figure_match:
                if current_paragraph:
                    blocks.append(self._create_text_block(
                        current_paragraph, page_num, doc_id, block_counter
                    ))
                    block_counter += 1
                    current_paragraph = []

                caption = figure_match.group(1).strip()
                blocks.append(IRBlock(
                    doc_id=doc_id,
                    page=page_num,
                    block_id=f"p{page_num}_b{block_counter}",
                    type='figure',
                    bbox=None,
                    reading_order=block_counter,
                    text=caption if caption else "[Figure]",
                    caption=caption,
                    parser_source='qwenvl'
                ))
                block_counter += 1
                continue

            # 3. Table detection
            if table_row_pattern.match(line_stripped):
                if not in_table:
                    # Start of table - flush paragraph
                    if current_paragraph:
                        blocks.append(self._create_text_block(
                            current_paragraph, page_num, doc_id, block_counter
                        ))
                        block_counter += 1
                        current_paragraph = []
                    in_table = True

                table_lines.append(line_stripped)
                continue
            elif in_table:
                # End of table
                if table_lines:
                    blocks.append(self._create_table_block(
                        table_lines, page_num, doc_id, block_counter
                    ))
                    block_counter += 1
                table_lines = []
                in_table = False
                # Fall through to process current line

            # 4. Formula block detection
            formula_match = formula_block_pattern.match(line_stripped)
            if formula_match:
                if current_paragraph:
                    blocks.append(self._create_text_block(
                        current_paragraph, page_num, doc_id, block_counter
                    ))
                    block_counter += 1
                    current_paragraph = []

                blocks.append(IRBlock(
                    doc_id=doc_id,
                    page=page_num,
                    block_id=f"p{page_num}_b{block_counter}",
                    type='formula',
                    bbox=None,
                    reading_order=block_counter,
                    text=formula_match.group(1).strip(),
                    markdown=line_stripped,
                    parser_source='qwenvl'
                ))
                block_counter += 1
                continue

            # 5. Regular text accumulation
            current_paragraph.append(line)

        # Flush remaining content
        if current_paragraph:
            blocks.append(self._create_text_block(
                current_paragraph, page_num, doc_id, block_counter
            ))
        if table_lines:
            blocks.append(self._create_table_block(
                table_lines, page_num, doc_id, block_counter
            ))

        return blocks

    def _create_text_block(
        self,
        lines: List[str],
        page_num: int,
        doc_id: str,
        block_id: int
    ) -> IRBlock:
        """Create text block from accumulated lines."""
        text = '\n'.join(lines).strip()
        return IRBlock(
            doc_id=doc_id,
            page=page_num,
            block_id=f"p{page_num}_b{block_id}",
            type='text',
            bbox=None,
            reading_order=block_id,
            text=text,
            markdown=text,
            parser_source='qwenvl'
        )

    def _create_table_block(
        self,
        table_lines: List[str],
        page_num: int,
        doc_id: str,
        block_id: int
    ) -> IRBlock:
        """Create table block from markdown table."""
        markdown_table = '\n'.join(table_lines)
        # Extract text content (remove | delimiters for plain text)
        text_lines = []
        for line in table_lines:
            # Skip separator line (|---|---|)
            if re.match(r'^\|[\s\-|]+\|$', line):
                continue
            # Extract cell values
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            text_lines.append(' | '.join(cells))

        return IRBlock(
            doc_id=doc_id,
            page=page_num,
            block_id=f"p{page_num}_b{block_id}",
            type='table',
            bbox=None,
            reading_order=block_id,
            text='\n'.join(text_lines),
            markdown=markdown_table,
            parser_source='qwenvl',
            raw_data={'table_format': 'markdown'}
        )

    @property
    def parser_name(self) -> str:
        return "qwenvl"

    def supports_tables(self) -> bool:
        return True

    def supports_bbox(self) -> bool:
        return False  # VLM doesn't provide pixel coordinates

    def supports_formulas(self) -> bool:
        return True
