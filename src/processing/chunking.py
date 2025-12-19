from typing import List, Dict

def chunk_with_layout_awareness(markdown: str, chunk_size=1000):
    """
    Chunk markdown content while preserving structure.
    - Respects Section headers (##)
    - Keeps Tables and Charts intact if possible
    - Merges small sections
    """
    import re
    
    lines = markdown.split('\n')
    chunks = []
    current_chunk = []
    current_section = "Introduction" # Default section
    token_count = 0
    
    # Simple whitespace tokenizer estimation
    def count_tokens(text):
        return len(text.split())

    for i, line in enumerate(lines):
        # Section Header Detection
        heading_match = re.match(r'^(#+)\s+(.+)$', line)
        
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            
            # If major section change (H1 or H2), force new chunk if current is substantial
            if level <= 2:
                if current_chunk and token_count > 100:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                    current_chunk = []
                    token_count = 0
                
                current_section = title
                current_chunk.append(line)
                token_count += count_tokens(line)
            else:
                # Sub-headers just append
                current_chunk.append(line)
                token_count += count_tokens(line)
        
        # Table Detection (Markdown table lines)
        elif line.strip().startswith('|'):
            # If start of table
            if not (current_chunk and current_chunk[-1].strip().startswith('|')):
                # Flush existing text if big enough
                if current_chunk and token_count > 500:
                     chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                     current_chunk = []
                     token_count = 0
            
            current_chunk.append(line)
            token_count += count_tokens(line)

        # Image/Chart placeholder
        elif line.strip().startswith('!['):
             current_chunk.append(line)
             token_count += count_tokens(line)

        else:
            current_chunk.append(line)
            token_count += count_tokens(line)
            
            # Check chunk size limit
            if token_count >= chunk_size:
                # Try to break at paragraph
                if not line.strip(): 
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'section': current_section,
                        'type': 'text'
                    })
                    current_chunk = []
                    token_count = 0
    
    # Final chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'section': current_section,
            'type': 'text'
        })
    
    return chunks
