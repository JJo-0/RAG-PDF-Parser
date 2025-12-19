"""
Deduplication module for RAG PDF Parser.
Detects duplicate PDFs, images, and URLs based on content hashing.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io


@dataclass
class DuplicateInfo:
    """Information about a detected duplicate."""
    hash: str
    original_path: str
    original_date: str
    duplicate_path: str
    duplicate_date: str
    type: str  # 'pdf', 'image', 'url'
    similarity: float = 1.0  # 1.0 = exact match


class Deduplicator:
    """
    Detects duplicates for PDFs, images, and URLs.
    Maintains a persistent hash database in JSON format.
    """

    def __init__(self, db_path: str = "output/.dedup_db.json"):
        """
        Initialize Deduplicator.

        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = db_path
        self.db = self._load_db()

    def _load_db(self) -> Dict:
        """Load existing database or create new one."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "pdfs": {},      # hash -> {path, date, size, pages}
            "images": {},    # hash -> {path, date, size, dimensions}
            "urls": {},      # hash -> {url, date, title}
            "texts": {}      # hash -> {path, date, preview}
        }

    def _save_db(self):
        """Save database to disk."""
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)

    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _compute_bytes_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of bytes."""
        return hashlib.sha256(data).hexdigest()

    def _compute_text_hash(self, text: str) -> str:
        """Compute hash of text content (normalized)."""
        # Normalize: lowercase, remove extra whitespace
        normalized = ' '.join(text.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _compute_image_hash(self, image_path: str) -> str:
        """
        Compute perceptual hash of image.
        Uses a simple average hash for speed.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize to 8x8
                img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                # Create binary hash
                bits = ''.join('1' if p > avg else '0' for p in pixels)
                return hex(int(bits, 2))[2:].zfill(16)
        except Exception:
            # Fallback to file hash
            return self._compute_file_hash(image_path)

    def _compute_url_hash(self, url: str) -> str:
        """Compute hash of URL (normalized)."""
        # Normalize URL: lowercase, remove trailing slash
        normalized = url.lower().rstrip('/')
        # Remove common tracking parameters
        if '?' in normalized:
            base, params = normalized.split('?', 1)
            # Keep only essential params
            normalized = base
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]

    def check_pdf(self, pdf_path: str) -> Optional[DuplicateInfo]:
        """
        Check if PDF is a duplicate.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DuplicateInfo if duplicate found, None otherwise
        """
        if not os.path.exists(pdf_path):
            return None

        file_hash = self._compute_file_hash(pdf_path)
        now = datetime.now().isoformat()

        if file_hash in self.db["pdfs"]:
            original = self.db["pdfs"][file_hash]
            return DuplicateInfo(
                hash=file_hash,
                original_path=original["path"],
                original_date=original["date"],
                duplicate_path=pdf_path,
                duplicate_date=now,
                type="pdf"
            )

        return None

    def register_pdf(self, pdf_path: str, pages: int = 0) -> str:
        """
        Register a PDF in the database.

        Returns:
            The file hash
        """
        file_hash = self._compute_file_hash(pdf_path)
        file_stat = os.stat(pdf_path)

        self.db["pdfs"][file_hash] = {
            "path": pdf_path,
            "date": datetime.now().isoformat(),
            "size": file_stat.st_size,
            "pages": pages,
            "filename": os.path.basename(pdf_path)
        }
        self._save_db()
        return file_hash

    def check_image(self, image_path: str, use_perceptual: bool = True) -> Optional[DuplicateInfo]:
        """
        Check if image is a duplicate.

        Args:
            image_path: Path to image file
            use_perceptual: Use perceptual hash (finds similar images)

        Returns:
            DuplicateInfo if duplicate found, None otherwise
        """
        if not os.path.exists(image_path):
            return None

        if use_perceptual:
            img_hash = self._compute_image_hash(image_path)
        else:
            img_hash = self._compute_file_hash(image_path)

        now = datetime.now().isoformat()

        if img_hash in self.db["images"]:
            original = self.db["images"][img_hash]
            return DuplicateInfo(
                hash=img_hash,
                original_path=original["path"],
                original_date=original["date"],
                duplicate_path=image_path,
                duplicate_date=now,
                type="image"
            )

        return None

    def register_image(self, image_path: str, use_perceptual: bool = True) -> str:
        """Register an image in the database."""
        if use_perceptual:
            img_hash = self._compute_image_hash(image_path)
        else:
            img_hash = self._compute_file_hash(image_path)

        try:
            with Image.open(image_path) as img:
                dimensions = f"{img.width}x{img.height}"
        except Exception:
            dimensions = "unknown"

        file_stat = os.stat(image_path)

        self.db["images"][img_hash] = {
            "path": image_path,
            "date": datetime.now().isoformat(),
            "size": file_stat.st_size,
            "dimensions": dimensions,
            "filename": os.path.basename(image_path)
        }
        self._save_db()
        return img_hash

    def check_url(self, url: str) -> Optional[DuplicateInfo]:
        """
        Check if URL was already processed.

        Args:
            url: URL string

        Returns:
            DuplicateInfo if duplicate found, None otherwise
        """
        url_hash = self._compute_url_hash(url)
        now = datetime.now().isoformat()

        if url_hash in self.db["urls"]:
            original = self.db["urls"][url_hash]
            return DuplicateInfo(
                hash=url_hash,
                original_path=original["url"],
                original_date=original["date"],
                duplicate_path=url,
                duplicate_date=now,
                type="url"
            )

        return None

    def register_url(self, url: str, title: str = "") -> str:
        """Register a URL in the database."""
        url_hash = self._compute_url_hash(url)

        self.db["urls"][url_hash] = {
            "url": url,
            "date": datetime.now().isoformat(),
            "title": title
        }
        self._save_db()
        return url_hash

    def check_text(self, text: str, preview_length: int = 100) -> Optional[DuplicateInfo]:
        """
        Check if text content is a duplicate.

        Args:
            text: Text content
            preview_length: Length of preview to store

        Returns:
            DuplicateInfo if duplicate found, None otherwise
        """
        text_hash = self._compute_text_hash(text)
        now = datetime.now().isoformat()

        if text_hash in self.db["texts"]:
            original = self.db["texts"][text_hash]
            return DuplicateInfo(
                hash=text_hash,
                original_path=original.get("path", ""),
                original_date=original["date"],
                duplicate_path="",
                duplicate_date=now,
                type="text"
            )

        return None

    def register_text(self, text: str, source_path: str = "", preview_length: int = 100) -> str:
        """Register text content in the database."""
        text_hash = self._compute_text_hash(text)

        self.db["texts"][text_hash] = {
            "path": source_path,
            "date": datetime.now().isoformat(),
            "preview": text[:preview_length] + "..." if len(text) > preview_length else text
        }
        self._save_db()
        return text_hash

    def check_all(self, file_path: str) -> Optional[DuplicateInfo]:
        """
        Check file against all types based on extension.

        Args:
            file_path: Path to file

        Returns:
            DuplicateInfo if duplicate found, None otherwise
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.check_pdf(file_path)
        elif ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}:
            return self.check_image(file_path)
        else:
            return None

    def get_stats(self) -> Dict:
        """Get statistics about the database."""
        return {
            "total_pdfs": len(self.db["pdfs"]),
            "total_images": len(self.db["images"]),
            "total_urls": len(self.db["urls"]),
            "total_texts": len(self.db["texts"]),
            "db_path": self.db_path
        }

    def get_all_entries(self, entry_type: str = None) -> List[Dict]:
        """
        Get all entries from database.

        Args:
            entry_type: 'pdfs', 'images', 'urls', 'texts', or None for all

        Returns:
            List of entries
        """
        if entry_type and entry_type in self.db:
            return [
                {"hash": k, "type": entry_type, **v}
                for k, v in self.db[entry_type].items()
            ]

        all_entries = []
        for etype in ["pdfs", "images", "urls", "texts"]:
            for k, v in self.db[etype].items():
                all_entries.append({"hash": k, "type": etype, **v})

        return sorted(all_entries, key=lambda x: x.get("date", ""), reverse=True)

    def clear_database(self):
        """Clear all entries from database."""
        self.db = {
            "pdfs": {},
            "images": {},
            "urls": {},
            "texts": {}
        }
        self._save_db()

    def remove_entry(self, hash_value: str, entry_type: str) -> bool:
        """Remove a specific entry from database."""
        if entry_type in self.db and hash_value in self.db[entry_type]:
            del self.db[entry_type][hash_value]
            self._save_db()
            return True
        return False
