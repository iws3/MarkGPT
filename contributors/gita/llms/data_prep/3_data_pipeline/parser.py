import os
import textwrap
import brotli
import warcio.bufferedreaders
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import trafilatura
import config

def patch_brotli():
    """Ensures that local modern Brotli libraries integrate smoothly with warcio's stream reader."""
    class SafeBrotliDecompressor:
        def __init__(self):
            self.decomp = brotli.Decompressor()
            self.unused_data = b""
        def decompress(self, data):
            return self.decomp.decompress(data)
            
    warcio.bufferedreaders.BufferedReader.DECOMPRESSORS['br'] = lambda: SafeBrotliDecompressor()

def wrap_output(text, width):
    """Wraps text output to avoid horizontal terminal scrolling."""
    wrapper = textwrap.TextWrapper(width=width, break_long_words=False, replace_whitespace=False)
    wrapped_lines = [wrapper.fill(line) for line in text.split('\n')]
    return '\n'.join(wrapped_lines)

def parse_uncompressed_warc(warc_path):
    print(f"[*] Parser: Reading uncompressed '{warc_path}' using main-content extraction...")
    patch_brotli()
    
    if not os.path.exists(warc_path):
        print(f"[!] Error: File '{warc_path}' does not exist. Run scraper.py first.")
        return

    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                html_content = record.content_stream().read()
                
                # Try extraction using FineWeb's choice: trafilatura
                cleaned_content = trafilatura.extract(
                    html_content, 
                    include_comments=False, 
                    include_tables=False
                )
                
                # Fallback to BeautifulSoup if the page doesn't have structured main article text
                if not cleaned_content:
                    soup = BeautifulSoup(html_content, "html.parser")
                    cleaned_content = soup.get_text(" ", strip=True) + " (Bs4 Fallback)"
                
                # Print with separation formatting
                print("=" * config.WRAP_WIDTH)
                print(f"URL: {url}")
                print("-" * config.WRAP_WIDTH)
                print(wrap_output(cleaned_content, config.WRAP_WIDTH))
                print("\n")

if __name__ == "__main__":
    parse_uncompressed_warc(config.WARC_FILE_NAME)
