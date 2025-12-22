"""
Persistent Cache module for RAG PDF Parser.

SQLite-backed caching for expensive operations (OCR, VLM, layout detection).
Significantly reduces reprocessing time for incremental updates.
"""

import os
import json
import sqlite3
import hashlib
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager


class PersistentCache:
    """
    SQLite-backed persistent cache for processing results.

    Caches:
    - OCR results (text + lines + confidence)
    - VLM captions (by image hash + prompt hash)
    - Layout detection results
    """

    def __init__(self, db_path: str = "output/.cache.db"):
        """
        Initialize the persistent cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # OCR results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    image_hash TEXT PRIMARY KEY,
                    text TEXT,
                    lines_json TEXT,
                    confidence REAL,
                    lang TEXT,
                    created_at TEXT,
                    hit_count INTEGER DEFAULT 0
                )
            """)

            # VLM caption cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vlm_cache (
                    cache_key TEXT PRIMARY KEY,
                    image_hash TEXT,
                    prompt_hash TEXT,
                    caption TEXT,
                    structured_json TEXT,
                    model TEXT,
                    created_at TEXT,
                    hit_count INTEGER DEFAULT 0
                )
            """)

            # Layout detection cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS layout_cache (
                    image_hash TEXT PRIMARY KEY,
                    blocks_json TEXT,
                    model TEXT,
                    created_at TEXT,
                    hit_count INTEGER DEFAULT 0
                )
            """)

            # Document processing status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doc_status (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT,
                    status TEXT,
                    pages_processed INTEGER,
                    total_pages INTEGER,
                    last_updated TEXT
                )
            """)

            # Create indices for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vlm_image ON vlm_cache(image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_path ON doc_status(source_path)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute hash of text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    # =========================================================================
    # OCR Cache Methods
    # =========================================================================

    def get_ocr(self, image_hash: str) -> Optional[Tuple[str, List[Dict], float, str]]:
        """
        Retrieve cached OCR result.

        Args:
            image_hash: Hash of the image

        Returns:
            Tuple of (text, lines, confidence, lang) or None if not cached
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT text, lines_json, confidence, lang
                FROM ocr_cache WHERE image_hash = ?
            """, (image_hash,))

            row = cursor.fetchone()
            if row:
                # Update hit count
                cursor.execute("""
                    UPDATE ocr_cache SET hit_count = hit_count + 1
                    WHERE image_hash = ?
                """, (image_hash,))
                conn.commit()

                lines = json.loads(row['lines_json']) if row['lines_json'] else []
                return (row['text'], lines, row['confidence'], row['lang'])

        return None

    def set_ocr(
        self,
        image_hash: str,
        text: str,
        lines: List[Dict],
        confidence: float,
        lang: str = "unknown"
    ):
        """
        Cache OCR result.

        Args:
            image_hash: Hash of the image
            text: Extracted text
            lines: Line-level OCR details
            confidence: Average confidence score
            lang: Detected language
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ocr_cache
                (image_hash, text, lines_json, confidence, lang, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                image_hash,
                text,
                json.dumps(lines, ensure_ascii=False),
                confidence,
                lang,
                datetime.now().isoformat()
            ))
            conn.commit()

    # =========================================================================
    # VLM Caption Cache Methods
    # =========================================================================

    def get_vlm_caption(
        self,
        image_hash: str,
        prompt_hash: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve cached VLM caption.

        Args:
            image_hash: Hash of the image
            prompt_hash: Hash of the prompt (for prompt-specific caching)

        Returns:
            Cached caption or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if prompt_hash:
                cache_key = f"{image_hash}_{prompt_hash}"
                cursor.execute("""
                    SELECT caption FROM vlm_cache WHERE cache_key = ?
                """, (cache_key,))
            else:
                cursor.execute("""
                    SELECT caption FROM vlm_cache WHERE image_hash = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (image_hash,))

            row = cursor.fetchone()
            if row:
                # Update hit count
                if prompt_hash:
                    cursor.execute("""
                        UPDATE vlm_cache SET hit_count = hit_count + 1
                        WHERE cache_key = ?
                    """, (f"{image_hash}_{prompt_hash}",))
                conn.commit()
                return row['caption']

        return None

    def set_vlm_caption(
        self,
        image_hash: str,
        caption: str,
        prompt_hash: Optional[str] = None,
        structured: Optional[Dict] = None,
        model: str = "unknown"
    ):
        """
        Cache VLM caption.

        Args:
            image_hash: Hash of the image
            caption: Generated caption
            prompt_hash: Hash of the prompt used
            structured: Structured caption data (if available)
            model: Model used for generation
        """
        cache_key = f"{image_hash}_{prompt_hash}" if prompt_hash else image_hash

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO vlm_cache
                (cache_key, image_hash, prompt_hash, caption, structured_json, model, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                cache_key,
                image_hash,
                prompt_hash or "",
                caption,
                json.dumps(structured, ensure_ascii=False) if structured else None,
                model,
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_vlm_structured(
        self,
        image_hash: str,
        prompt_hash: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Retrieve cached structured VLM output.

        Returns:
            Parsed structured data or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if prompt_hash:
                cache_key = f"{image_hash}_{prompt_hash}"
                cursor.execute("""
                    SELECT structured_json FROM vlm_cache WHERE cache_key = ?
                """, (cache_key,))
            else:
                cursor.execute("""
                    SELECT structured_json FROM vlm_cache WHERE image_hash = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (image_hash,))

            row = cursor.fetchone()
            if row and row['structured_json']:
                return json.loads(row['structured_json'])

        return None

    # =========================================================================
    # Layout Cache Methods
    # =========================================================================

    def get_layout(self, image_hash: str) -> Optional[List[Dict]]:
        """
        Retrieve cached layout detection result.

        Args:
            image_hash: Hash of the page image

        Returns:
            List of block dictionaries or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT blocks_json FROM layout_cache WHERE image_hash = ?
            """, (image_hash,))

            row = cursor.fetchone()
            if row:
                cursor.execute("""
                    UPDATE layout_cache SET hit_count = hit_count + 1
                    WHERE image_hash = ?
                """, (image_hash,))
                conn.commit()
                return json.loads(row['blocks_json'])

        return None

    def set_layout(
        self,
        image_hash: str,
        blocks: List[Dict],
        model: str = "surya"
    ):
        """
        Cache layout detection result.

        Args:
            image_hash: Hash of the page image
            blocks: List of detected blocks
            model: Model used for detection
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO layout_cache
                (image_hash, blocks_json, model, created_at, hit_count)
                VALUES (?, ?, ?, ?, 0)
            """, (
                image_hash,
                json.dumps(blocks, ensure_ascii=False),
                model,
                datetime.now().isoformat()
            ))
            conn.commit()

    # =========================================================================
    # Document Status Methods
    # =========================================================================

    def get_doc_status(self, doc_id: str) -> Optional[Dict]:
        """Get processing status for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM doc_status WHERE doc_id = ?
            """, (doc_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)

        return None

    def set_doc_status(
        self,
        doc_id: str,
        source_path: str,
        status: str,
        pages_processed: int,
        total_pages: int
    ):
        """Update processing status for a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO doc_status
                (doc_id, source_path, status, pages_processed, total_pages, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                source_path,
                status,
                pages_processed,
                total_pages,
                datetime.now().isoformat()
            ))
            conn.commit()

    # =========================================================================
    # Cache Management Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM ocr_cache")
            row = cursor.fetchone()
            stats['ocr'] = {'count': row[0], 'hits': row[1] or 0}

            cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM vlm_cache")
            row = cursor.fetchone()
            stats['vlm'] = {'count': row[0], 'hits': row[1] or 0}

            cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM layout_cache")
            row = cursor.fetchone()
            stats['layout'] = {'count': row[0], 'hits': row[1] or 0}

            cursor.execute("SELECT COUNT(*) FROM doc_status")
            stats['documents'] = cursor.fetchone()[0]

            return stats

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            cache_type: Specific cache to clear ('ocr', 'vlm', 'layout', 'docs')
                       or None to clear all
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if cache_type == 'ocr' or cache_type is None:
                cursor.execute("DELETE FROM ocr_cache")
            if cache_type == 'vlm' or cache_type is None:
                cursor.execute("DELETE FROM vlm_cache")
            if cache_type == 'layout' or cache_type is None:
                cursor.execute("DELETE FROM layout_cache")
            if cache_type == 'docs' or cache_type is None:
                cursor.execute("DELETE FROM doc_status")

            conn.commit()

    def vacuum(self):
        """Reclaim unused space in the database."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
