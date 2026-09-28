from io import BytesIO
from datetime import datetime
import requests
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders
import config

def fetch_and_write_to_warc(warc_path, target_urls):
    print(f"[*] Scraper: Fetching websites and building uncompressed '{warc_path}'...")
    
    # Open the file in write-binary mode WITHOUT GZIP compression
    with open(warc_path, 'wb') as output_file:
        # gzip=False disables compression entirely
        writer = WARCWriter(output_file, gzip=False)

        for url in target_urls:
            try:
                print(f"    -> Scraping: {url}")
                response = requests.get(url, timeout=config.TIMEOUT)
                response.raise_for_status()

                # Format response headers
                http_headers_list = [(k, v) for k, v in response.headers.items()]
                http_headers = StatusAndHeaders(
                    f"{response.status_code} {response.reason}",
                    http_headers_list,
                    protocol='HTTP/1.1'
                )

                # Setup WARC headers
                warc_date = response.headers.get('Date') or datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                warc_headers_dict = {
                    'WARC-Date': warc_date,
                    'WARC-Target-URI': url
                }

                # Create raw WARC record
                record = writer.create_warc_record(
                    uri=url,
                    record_type='response',
                    payload=BytesIO(response.content),
                    http_headers=http_headers,
                    warc_headers_dict=warc_headers_dict
                )
                
                writer.write_record(record)
                print(f"    [+] Added {url} to the uncompressed WARC.")

            except Exception as e:
                print(f"    [!] Scraping failed for {url}: {e}")

if __name__ == "__main__":
    fetch_and_write_to_warc(config.WARC_FILE_NAME, config.URLS)
