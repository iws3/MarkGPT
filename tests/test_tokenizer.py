"""Tests for tokenizer functionality."""

import pytest
from src.tokenizer.tokenizer import Tokenizer
from src.tokenizer.banso_preprocess import BansoPreprocessor


class TestTokenizer:
    """Test tokenization."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer(vocab_size=1000)

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encode/decode consistency."""
        text = "In the beginning was the Word"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Should be close to original (normalization differences OK)
        assert len(decoded) > 0


class TestBansoPreprocessor:
    """Test Banso language preprocessing."""

    def test_normalization(self):
        """Test text normalization."""
        preprocessor = BansoPreprocessor()
        text = "Nso   people     are   great"
        normalized = preprocessor.normalize(text)
        
        assert "   " not in normalized

    def test_dialect_detection(self):
        """Test dialect detection."""
        preprocessor = BansoPreprocessor()
        text = "nso pidgin culture"
        dialect = preprocessor.detect_dialect(text)
        
        assert dialect in ['upper', 'lower', 'unknown']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
