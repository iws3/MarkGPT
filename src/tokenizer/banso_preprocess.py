"""Lamnso' (Nso') language preprocessing utilities.

Handles Nso'-specific normalization, dialect detection, and tone restoration.
"""

import re
from typing import Optional


class BansoPreprocessor:
    """Preprocess Lamnso' text for tokenization.
    
    Handles dialect-specific normalizations and linguistic features.
    """

    def __init__(self):
        """Initialize preprocessor."""
        # Dialect markers for detecting Nso' variations
        self.dialect_markers = {
            'upper_nso': ['nso', 'pidgin'],
            'lower_nso': ['fontem', 'menumba'],
        }
        
        # Common words for tone detection
        self.tone_words = {
            'sori': 'sórì',  # Sorry (low-high-high)
            'tuma': 'túmà',  # Send
        }

    def detect_dialect(self, text: str) -> str:
        """Detect Nso' dialect from vocabulary markers.
        
        Args:
            text: Input text
        
        Returns:
            Detected dialect ('upper', 'lower', 'unknown')
        """
        tokens = text.lower().split()
        
        upper_count = sum(1 for t in tokens if t in self.dialect_markers['upper_nso'])
        lower_count = sum(1 for t in tokens if t in self.dialect_markers['lower_nso'])
        
        if upper_count > lower_count:
            return 'upper'
        elif lower_count > upper_count:
            return 'lower'
        else:
            return 'unknown'

    def normalize(self, text: str) -> str:
        """Normalize Nso' text.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Standardize punctuation
        text = re.sub(r'["""]', '"', text)  # Normalize quotes
        
        return text

    def restore_tone(self, text: str) -> str:
        """Attempt to restore tonal marks based on dictionary.
        
        Args:
            text: Text to process
        
        Returns:
            Text with restored tones where possible
        """
        for word, toned in self.tone_words.items():
            text = re.sub(r'\b' + word + r'\b', toned, text, flags=re.IGNORECASE)
        
        return text
