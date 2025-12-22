"""
Simple test script for RAG PDF Parser IR Pipeline.

Tests basic functionality without full processing.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.models.block import IRBlock, IRPage, IRDocument
        from src.models.chunk import IRChunk, ChunkingConfig
        from src.config import ProcessorConfig
        from src.processing.ir_processor import IRPipelineProcessor
        from src.output.writer import OutputWriter
        from src.cache.persistent import PersistentCache
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_data_models():
    """Test IR data models."""
    print("\nTesting data models...")

    try:
        from src.models.block import IRBlock, IRPage, IRDocument
        from src.models.chunk import IRChunk

        # Test IRBlock creation
        block = IRBlock(
            doc_id="test123",
            page=1,
            block_id="p1_b0",
            type="text",
            bbox=[0, 0, 100, 100],
            reading_order=0,
            text="Test text",
            confidence=0.95
        )

        assert block.anchor == "[@p1_txt0]", "Anchor generation failed"
        assert block.to_dict()['doc_id'] == "test123", "to_dict failed"

        # Test IRDocument
        doc = IRDocument(
            doc_id="test123",
            source_path="/test/path.pdf",
            filename="path.pdf",
            total_pages=1
        )

        assert doc.created_at is not None, "Timestamp not set"

        # Test IRChunk
        chunk = IRChunk(
            chunk_id="test_c0",
            doc_id="test123",
            text="Chunk text"
        )

        assert chunk.char_count > 0, "Char count not computed"

        print("  [OK] Data models working correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Data model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    try:
        from src.config import ProcessorConfig

        # Test default config
        config = ProcessorConfig()
        assert config.output_mode == "markdown"
        assert config.dpi == 200

        # Test RAG preset
        rag_config = ProcessorConfig.for_rag_pipeline()
        assert rag_config.output_mode == "both"
        assert rag_config.enable_chunking == True

        print("  [OK] Configuration working correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Config test failed: {e}")
        return False


def test_cache():
    """Test persistent cache."""
    print("\nTesting persistent cache...")

    try:
        from src.cache.persistent import PersistentCache
        import tempfile

        # Use temp file for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tf:
            db_path = tf.name

        cache = PersistentCache(db_path=db_path)

        # Test OCR cache
        cache.set_ocr("hash123", "test text", [], 0.95, "en")
        result = cache.get_ocr("hash123")
        assert result is not None, "OCR cache failed"
        assert result[0] == "test text", "OCR cache data mismatch"

        # Test VLM cache
        cache.set_vlm_caption("hash456", "test caption", model="test")
        caption = cache.get_vlm_caption("hash456")
        assert caption == "test caption", "VLM cache failed"

        # Test stats
        stats = cache.get_stats()
        assert stats['ocr']['count'] > 0, "Stats failed"

        # Cleanup
        os.unlink(db_path)

        print("  [OK] Cache working correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_writer():
    """Test output writer."""
    print("\nTesting output writer...")

    try:
        from src.output.writer import OutputWriter
        from src.models.block import IRDocument, IRPage, IRBlock
        import tempfile
        import os

        # Create temp output dir
        temp_dir = tempfile.mkdtemp()

        writer = OutputWriter(temp_dir)

        # Create test document
        doc = IRDocument(
            doc_id="test123",
            source_path="/test/path.pdf",
            filename="test.pdf",
            total_pages=1
        )

        page = IRPage(
            doc_id="test123",
            page_num=1,
            width=600,
            height=800,
            dpi=200
        )

        block = IRBlock(
            doc_id="test123",
            page=1,
            block_id="p1_b0",
            type="text",
            bbox=[0, 0, 100, 100],
            reading_order=0,
            text="Test content",
            markdown="Test content",
            confidence=0.95
        )

        page.blocks.append(block)
        doc.pages.append(page)

        # Test markdown output
        md_path = writer.write_markdown(doc, with_anchors=True)
        assert os.path.exists(md_path), "Markdown file not created"

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test content" in content, "Content not in markdown"

        # Test JSONL output
        jsonl_path = writer.write_ir_jsonl(doc)
        assert os.path.exists(jsonl_path), "JSONL file not created"

        # Test metadata
        meta_path = writer.write_metadata(doc)
        assert os.path.exists(meta_path), "Metadata file not created"

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print("  [OK] Output writer working correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Output writer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunking():
    """Test IR-aware chunking."""
    print("\nTesting chunking...")

    try:
        from src.processing.chunking import chunk_with_ir_awareness
        from src.models.block import IRBlock
        from src.models.chunk import ChunkingConfig

        # Create test blocks
        blocks = []
        for i in range(5):
            block = IRBlock(
                doc_id="test123",
                page=1,
                block_id=f"p1_b{i}",
                type="text",
                bbox=[0, i*100, 100, (i+1)*100],
                reading_order=i,
                text=f"Test paragraph {i}. " * 50,  # ~500 chars each
                confidence=0.95
            )
            blocks.append(block)

        # Test chunking
        config = ChunkingConfig(chunk_size=500, overlap_tokens=50)
        chunks = chunk_with_ir_awareness(blocks, config)

        assert len(chunks) > 0, "No chunks created"
        assert chunks[0].doc_id == "test123", "Doc ID not preserved"
        assert len(chunks[0].block_ids) > 0, "Block IDs not tracked"

        print(f"  [OK] Chunking working correctly ({len(chunks)} chunks created)")
        return True
    except Exception as e:
        print(f"  [FAIL] Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RAG PDF Parser - Pipeline Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_data_models,
        test_config,
        test_cache,
        test_output_writer,
        test_chunking
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("[OK] All tests passed!")
        return 0
    else:
        print(f"[FAIL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
