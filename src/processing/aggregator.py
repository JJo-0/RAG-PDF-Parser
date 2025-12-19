from ..layout.detector import LayoutDetector
from ..text.extractor import TextExtractor
from ..table.extractor import TableExtractor
from ..captioning import ImageCaptioner
from .heading import HeadingDetector
from PIL import Image
import os
import uuid
from typing import List, Dict, Tuple, Optional

class MarkdownAggregator:
    def __init__(self, vlm_concurrency=3):
        print("Loading Markdown Aggregator...")
        # Initialize sub-modules
        self.layout_detector = LayoutDetector()
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()
        self.heading_detector = HeadingDetector()
        self.captioner = ImageCaptioner(max_concurrent=vlm_concurrency)

    def process_page(self, page, page_idx=1):
        """
        Process a single page (fitz.Page) and return Markdown content.
        Optimized with batch OCR and async VLM captioning.
        """
        # Generate Image for Visual Processing
        pix = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 1. Detect Layout
        print("  - Detecting layout...")
        blocks = self.layout_detector.detect(image)

        # 2. Sort blocks by reading order
        blocks.sort(key=lambda x: x.get('order') if x.get('order') is not None else float('inf'))

        # Check if we have any meaningful content blocks
        content_labels = ['Text', 'Section-header', 'Title', 'ListItem', 'Caption', 'Table', 'Picture', 'Chart', 'Figure', 'Formula', 'SectionHeader']
        has_text_content = any(block['label'] in content_labels for block in blocks)

        # Fallback Logic:
        if not has_text_content:
            return self._process_fallback(page, image, page_idx)

        return self._process_blocks_optimized(image, blocks, page_idx)

    def _process_blocks_optimized(self, image: Image.Image, blocks: List[Dict], page_idx: int) -> str:
        """
        Optimized block processing with batch OCR and parallel VLM.
        """
        print(f"  - Processing {len(blocks)} blocks (optimized)...")

        # Separate blocks by type
        text_labels = ['Text', 'Section-header', 'Title', 'ListItem', 'Caption', 'Footer', 'Header', 'PageHeader', 'PageFooter', 'SectionHeader']
        image_labels = ['Picture', 'Chart', 'Figure', 'Formula', 'Table']

        text_blocks = [(i, b) for i, b in enumerate(blocks) if b['label'] in text_labels]
        image_blocks = [(i, b) for i, b in enumerate(blocks) if b['label'] in image_labels]

        # Results storage (indexed by original position)
        results = {i: None for i in range(len(blocks))}

        # === BATCH OCR for text blocks ===
        if text_blocks:
            text_bboxes = [b['bbox'] for _, b in text_blocks]
            print(f"    - Batch OCR for {len(text_bboxes)} text regions...")
            ocr_results = self.text_extractor.extract_text_batch(image, text_bboxes)

            for (orig_idx, block), (text, _) in zip(text_blocks, ocr_results):
                level = self.heading_detector.detect({'label': block['label'], 'text': text, 'bbox': block['bbox']})

                if level == 1:
                    content = f"# {text}"
                elif level == 2:
                    content = f"## {text}"
                elif level == 3:
                    content = f"### {text}"
                else:
                    content = text

                results[orig_idx] = content

        # === BATCH VLM Captioning for images ===
        if image_blocks:
            print(f"    - Processing {len(image_blocks)} images with batch captioning...")

            # Crop and save all images first (collect PIL images for direct captioning)
            image_data = []  # (orig_idx, label, filename, crop_image)
            for orig_idx, block in image_blocks:
                label = block['label']
                bbox = block['bbox']
                filename, crop_img = self._save_crop_and_return(image, bbox, label, page_idx)
                image_data.append((orig_idx, label, filename, crop_img))

            # Batch caption generation (pass PIL images directly to avoid re-reading)
            crop_images = [data[3] for data in image_data]
            print(f"    - Generating {len(crop_images)} captions in parallel...")
            captions = self.captioner.caption_batch(crop_images)

            # Build markdown for each image
            for (orig_idx, label, filename, _), caption in zip(image_data, captions):
                content = f"\n![{label}]({filename})\n<!-- {label} detected -->\n"
                if caption:
                    if label == 'Table':
                        content += f"\n*AI Table Summary: {caption}*\n"
                    else:
                        content += f"\n*AI Caption: {caption}*\n"
                results[orig_idx] = content

        # Assemble in original order
        markdown_output = []
        for i in range(len(blocks)):
            if results[i]:
                markdown_output.append(results[i])
                markdown_output.append("")

        return "\n".join(markdown_output)

    def _process_fallback(self, page, image: Image.Image, page_idx: int) -> str:
        """Hybrid fallback when layout detection fails."""
        print("  ! No layout content detected. Using Hybrid Fallback (OCR + Native Images)...")

        markdown_output = []

        # A. Native Image Extraction (with batch captioning)
        native_images = self.extract_native_images_optimized(page, page_idx)

        # B. Whole Page OCR
        text_result, _ = self.text_extractor.extract_text(image)

        if text_result:
            markdown_output.append(text_result)

        if native_images:
            markdown_output.append("\n<!-- Detected Figures/Images (Fallback) -->\n")
            for img_path, _, caption in native_images:
                content = f"\n![Figure]({img_path})\n"
                if caption:
                    content += f"\n*AI Caption: {caption}*\n"
                markdown_output.append(content)

        return "\n".join(markdown_output)

    def _save_crop_and_return(self, image: Image.Image, bbox: List[float], label: str, page_idx: int) -> Tuple[str, Image.Image]:
        """
        Crop and save image, returning both filename and PIL Image.
        Avoids re-reading from disk for captioning.
        """
        output_dir = getattr(self, 'output_dir', 'output')
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        try:
            crop = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{label}_{page_idx}_{unique_id}.png"
            path = os.path.join(images_dir, filename)
            crop.save(path)
            return f"images/{filename}", crop
        except Exception as e:
            print(f"Error saving crop: {e}")
            return "placeholder", Image.new('RGB', (100, 100))

    def extract_native_images(self, page, page_idx):
        """
        Extract images directly from PDF page using PyMuPDF.
        Returns list of (relative_path, bbox, caption).
        Legacy method - use extract_native_images_optimized for better performance.
        """
        results = []
        image_list = page.get_images(full=True)

        output_dir = getattr(self, 'output_dir', 'output')
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            rects = page.get_image_rects(xref)

            if not rects:
                continue

            bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1]

            if (bbox[2] - bbox[0]) < 50 or (bbox[3] - bbox[1]) < 50:
                continue

            unique_id = uuid.uuid4().hex[:8]
            filename = f"native_figure_{page_idx}_{img_index}_{unique_id}.{ext}"
            path = os.path.join(images_dir, filename)

            with open(path, "wb") as f:
                f.write(image_bytes)

            print(f"    - Generating caption for native image {filename}...")
            caption = self.captioner.caption(path)

            results.append((f"images/{filename}", bbox, caption))

        return results

    def extract_native_images_optimized(self, page, page_idx) -> List[Tuple[str, List[float], Optional[str]]]:
        """
        Optimized native image extraction with batch captioning.
        Returns list of (relative_path, bbox, caption).
        """
        from io import BytesIO

        output_dir = getattr(self, 'output_dir', 'output')
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        image_list = page.get_images(full=True)

        # First pass: extract and save all images, collect PIL images for captioning
        extracted = []  # (relative_path, bbox, pil_image)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1]

                # Filter small icons
                if (bbox[2] - bbox[0]) < 50 or (bbox[3] - bbox[1]) < 50:
                    continue

                # Save to disk
                unique_id = uuid.uuid4().hex[:8]
                filename = f"native_figure_{page_idx}_{img_index}_{unique_id}.{ext}"
                path = os.path.join(images_dir, filename)

                with open(path, "wb") as f:
                    f.write(image_bytes)

                # Load as PIL for captioning (avoid re-reading)
                pil_img = Image.open(BytesIO(image_bytes))

                extracted.append((f"images/{filename}", bbox, pil_img))

            except Exception as e:
                print(f"    ! Error extracting native image {img_index}: {e}")
                continue

        if not extracted:
            return []

        # Batch caption all images
        pil_images = [item[2] for item in extracted]
        print(f"    - Batch captioning {len(pil_images)} native images...")
        captions = self.captioner.caption_batch(pil_images)

        # Combine results
        results = []
        for (rel_path, bbox, _), caption in zip(extracted, captions):
            results.append((rel_path, bbox, caption))

        return results

    def save_crop(self, image, bbox, label, page_idx):
        """
        Crop the bbox from image and save to disk.
        Returns relative path for markdown.
        """
        import os
        import uuid
        
        output_dir = getattr(self, 'output_dir', 'output')
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Crop
        try:
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{label}_{page_idx}_{unique_id}.png"
            path = os.path.join(images_dir, filename)
            crop.save(path)
            
            return f"images/{filename}"
        except Exception as e:
            print(f"Error saving crop: {e}")
            return "placeholder"

    def aggregate(self, doc, output_dir="output"):
        """
        Process PDF Document.
        """
        self.output_dir = output_dir 
        full_markdown = ""
        
        for i, page in enumerate(doc):
            print(f"Processing Page {i+1}/{len(doc)}...")
            page_md = self.process_page(page, page_idx=i+1)
            full_markdown += f"\n\n<!-- Page {i+1} -->\n\n"
            full_markdown += page_md
            
        return full_markdown
