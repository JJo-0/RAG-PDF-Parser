"""
RAG PDF Parser - Markdown Viewer with Translation & Deduplication
Streamlit app to preview parsed markdown results with images, bilingual translation, and duplicate detection.

Usage:
    python -m streamlit run streamlit_viewer.py
    python -m streamlit run streamlit_viewer.py -- --output_dir custom_output
"""

import streamlit as st
import os
import re
import argparse
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.translation import Translator
from src.dedup import Deduplicator

# Page config
st.set_page_config(
    page_title="RAG PDF Viewer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_translator():
    """Cache translator instance."""
    return Translator(model="gpt-oss:20b", max_concurrent=2)

@st.cache_resource
def get_deduplicator():
    """Cache deduplicator instance."""
    return Deduplicator(db_path="output/.dedup_db.json")

def get_output_dir():
    """Get output directory from command line args or default."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    try:
        args, _ = parser.parse_known_args()
        return args.output_dir
    except:
        return "output"

def get_md_files(output_dir: str) -> list:
    """Get list of markdown files in output directory."""
    if not os.path.exists(output_dir):
        return []

    md_files = []
    for f in os.listdir(output_dir):
        if f.endswith('.md'):
            md_files.append(f)

    return sorted(md_files)

def get_images(output_dir: str) -> list:
    """Get list of images in the images subdirectory."""
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        return []

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    images = []

    for f in os.listdir(images_dir):
        if Path(f).suffix.lower() in image_extensions:
            images.append(f)

    return sorted(images)

def fix_image_paths(content: str, output_dir: str) -> str:
    """Convert relative image paths to absolute paths for Streamlit rendering."""
    def replace_path(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        if img_path.startswith('./'):
            img_path = img_path[2:]

        abs_path = os.path.join(output_dir, img_path)

        if os.path.exists(abs_path):
            return f'![{alt_text}]({abs_path})'
        return match.group(0)

    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    return re.sub(pattern, replace_path, content)

def render_markdown_with_images(content: str, output_dir: str):
    """Render markdown content with proper image handling."""
    content = fix_image_paths(content, output_dir)

    parts = re.split(r'(!\[[^\]]*\]\([^)]+\))', content)

    for part in parts:
        if not part.strip():
            continue

        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', part)

        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)

            if os.path.exists(img_path):
                st.image(img_path, caption=alt_text if alt_text else None, use_container_width=True)
            else:
                st.warning(f"Image not found: {img_path}")
        else:
            st.markdown(part, unsafe_allow_html=True)

def translate_content_with_progress(content: str, target_lang: str) -> str:
    """Translate content paragraph by paragraph with progress bar."""
    translator = get_translator()

    # Parse paragraphs
    paragraphs = translator._parse_markdown_paragraphs(content)

    # Count translatable paragraphs
    text_paragraphs = [(i, p) for i, (ptype, p) in enumerate(paragraphs) if ptype == 'text' and p.strip()]

    if not text_paragraphs:
        return content

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    translations = {}
    total = len(text_paragraphs)

    for idx, (para_idx, para_text) in enumerate(text_paragraphs):
        status_text.text(f"Translating paragraph {idx + 1}/{total}...")
        progress_bar.progress((idx + 1) / total)

        translation = translator.translate(para_text, target_lang=target_lang)
        translations[para_idx] = translation

    # Build result
    result_lines = []
    for i, (ptype, content_part) in enumerate(paragraphs):
        if ptype == 'special':
            result_lines.append(content_part)
        else:
            if content_part.strip() and i in translations:
                result_lines.append(content_part)
                result_lines.append('')
                if translations[i]:
                    result_lines.append(f"> *{translations[i]}*")
                else:
                    result_lines.append("> *(translation unavailable)*")
                result_lines.append('')
            else:
                result_lines.append(content_part)

    progress_bar.empty()
    status_text.empty()

    return '\n'.join(result_lines)

def show_dedup_page():
    """Show deduplication management page."""
    st.title("🔍 Duplicate Detection")

    dedup = get_deduplicator()
    stats = dedup.get_stats()

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PDFs", stats["total_pdfs"])
    with col2:
        st.metric("Images", stats["total_images"])
    with col3:
        st.metric("URLs", stats["total_urls"])
    with col4:
        st.metric("Texts", stats["total_texts"])

    st.divider()

    # File upload for duplicate check
    st.subheader("Check for Duplicates")

    uploaded_file = st.file_uploader(
        "Upload a file to check",
        type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp']
    )

    if uploaded_file:
        # Save temporarily
        temp_path = f"output/.temp_{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Check for duplicate
        dup_info = dedup.check_all(temp_path)

        if dup_info:
            st.error(f"⚠️ Duplicate detected!")
            st.json({
                "type": dup_info.type,
                "original_file": dup_info.original_path,
                "original_date": dup_info.original_date,
                "hash": dup_info.hash[:16] + "..."
            })
        else:
            st.success("✅ No duplicate found!")

            if st.button("Register this file"):
                if uploaded_file.name.lower().endswith('.pdf'):
                    dedup.register_pdf(temp_path)
                else:
                    dedup.register_image(temp_path)
                st.success("File registered!")
                st.rerun()

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # URL check
    st.subheader("Check URL")
    url_input = st.text_input("Enter URL to check")

    if url_input:
        dup_info = dedup.check_url(url_input)

        if dup_info:
            st.error(f"⚠️ URL already processed!")
            st.json({
                "original_url": dup_info.original_path,
                "original_date": dup_info.original_date
            })
        else:
            st.success("✅ New URL!")
            if st.button("Register URL"):
                dedup.register_url(url_input)
                st.success("URL registered!")
                st.rerun()

    st.divider()

    # Database viewer
    st.subheader("Database Entries")

    entry_type = st.selectbox(
        "Filter by type",
        ["All", "pdfs", "images", "urls", "texts"]
    )

    entries = dedup.get_all_entries(None if entry_type == "All" else entry_type)

    if entries:
        for entry in entries[:50]:  # Limit display
            with st.expander(f"{entry.get('filename', entry.get('url', entry.get('preview', 'Entry')[:30]))}"):
                st.json(entry)
    else:
        st.info("No entries in database")

    # Clear database button
    st.divider()
    if st.button("🗑️ Clear Database", type="secondary"):
        dedup.clear_database()
        st.success("Database cleared!")
        st.rerun()

def show_viewer_page():
    """Show main viewer page."""
    output_dir = get_output_dir()

    md_files = get_md_files(output_dir)
    images = get_images(output_dir)

    if not md_files:
        st.warning(f"No markdown files found in `{output_dir}/`")
        st.info("Run the parser first:\n```\npython main.py your.pdf\n```")
        return

    # Sidebar content
    with st.sidebar:
        st.subheader("📑 Documents")

        selected_file = st.radio(
            "Select a document:",
            md_files,
            format_func=lambda x: f"📄 {x[:-3]}"
        )

        st.divider()

        # Translation settings
        st.subheader("🌐 Translation")

        enable_translation = st.toggle("Enable Translation", value=False)

        if enable_translation:
            target_lang = st.radio(
                "Translate to:",
                ["ko", "en"],
                format_func=lambda x: "🇰🇷 Korean" if x == "ko" else "🇺🇸 English",
                horizontal=True
            )
            st.caption(f"Model: `gpt-oss:20b`")

        st.divider()

        # Image gallery
        st.subheader(f"🖼️ Images ({len(images)})")

        if images:
            with st.expander("View all images", expanded=False):
                for img in images:
                    img_path = os.path.join(output_dir, "images", img)
                    st.image(img_path, caption=img, use_container_width=True)
        else:
            st.caption("No images extracted")

    # Main content
    if selected_file:
        file_path = os.path.join(output_dir, selected_file)

        st.markdown(f'## 📄 {selected_file[:-3]}')

        # File info
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size / 1024

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        img_count = len(re.findall(r'!\[[^\]]*\]\([^)]+\)', content))
        page_count = len(re.findall(r'<!-- Page \d+ -->', content))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("File Size", f"{file_size:.1f} KB")
        with col2:
            st.metric("Images", img_count)
        with col3:
            st.metric("Pages", page_count if page_count > 0 else "N/A")
        with col4:
            if enable_translation:
                st.metric("Mode", "Bilingual")
            else:
                st.metric("Mode", "Original")

        st.divider()

        # View mode
        view_mode = st.radio(
            "View Mode",
            ["Rendered", "Raw Markdown"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Apply translation if enabled
        display_content = content
        if enable_translation:
            cache_key = f"trans_{selected_file}_{target_lang}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = translate_content_with_progress(content, target_lang)
            display_content = st.session_state[cache_key]

        # Content display
        if view_mode == "Raw Markdown":
            st.code(display_content, language="markdown")
        else:
            render_markdown_with_images(display_content, output_dir)

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    blockquote {
        border-left: 3px solid #4CAF50;
        padding-left: 1rem;
        margin: 0.5rem 0;
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 0.5rem 1rem;
        border-radius: 0 4px 4px 0;
    }
    .stImage {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 4px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.title("📄 RAG PDF Viewer")
        st.divider()

        page = st.radio(
            "Navigation",
            ["📖 Viewer", "🔍 Duplicates"],
            label_visibility="collapsed"
        )

        st.divider()
        st.caption(f"📁 Output: `{get_output_dir()}`")

        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                if key.startswith("trans_"):
                    del st.session_state[key]
            st.rerun()

    # Page routing
    if page == "📖 Viewer":
        show_viewer_page()
    else:
        show_dedup_page()

if __name__ == "__main__":
    main()
