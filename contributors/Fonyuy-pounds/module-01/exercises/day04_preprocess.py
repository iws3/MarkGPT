import re
from typing import List

def load_and_preprocess(text: str) -> List[str]:
    """
    Convert raw text into a clean list of lowercase word tokens.
    """
    tokens = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Step 1: Convert to lowercase
        line = line.lower()
        
        # Step 2: Remove punctuation
        line = re.sub(r'[^\w\s]', '', line)
        
        # Step 3: Split on whitespace
        words = line.split()
        
        # Step 4: Filter out empty strings
        words = [w for w in words if w]
        
        # Step 5: Add tokens and <EOS>
        tokens.extend(words)
        tokens.append('<EOS>')
    
    return tokens
