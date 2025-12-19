import difflib
import sys
import re

def clean_text(text):
    # Remove markdown tags, special chars, extra whitespace for fair comparison
    # Keep only alphanumeric and Korean characters
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compare_files(ground_truth_path, generated_path):
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_text = f.read()
            
        with open(generated_path, 'r', encoding='utf-8') as f:
            gen_text = f.read()
            
        clean_gt = clean_text(gt_text)
        clean_gen = clean_text(gen_text)
        
        matcher = difflib.SequenceMatcher(None, clean_gt, clean_gen)
        ratio = matcher.ratio()
        
        print(f"Similarity Score: {ratio:.4f}")
        print(f"Ground Truth Length (cleaned): {len(clean_gt)}")
        print(f"Generated Length (cleaned): {len(clean_gen)}")
        
        # Identify longest matching blocks
        match = matcher.find_longest_match(0, len(clean_gt), 0, len(clean_gen))
        print(f"Longest common substring length: {match.size}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_text.py <ground_truth> <generated>")
        sys.exit(1)
        
    compare_files(sys.argv[1], sys.argv[2])
