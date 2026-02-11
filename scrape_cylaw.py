"""
CyLaw.org Scraper — Downloads all available legal PDFs from the Cypriot law database.

Phases:
1. Discover all index pages from /nomoi/indexes/
2. Extract PDF links from each index page
3. Download PDFs to ~/test_files_cy/
4. Fallback: scrape HTML full text for laws without PDFs

Usage:
    python scrape_cylaw.py
    python scrape_cylaw.py --output-dir /path/to/output
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9,el;q=0.8",
    "Referer": "https://www.cylaw.org/",
}


def _make_session() -> requests.Session:
    """Create a session with browser headers and retry backoff."""
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.cylaw.org"
INDEX_DIR_URL = f"{BASE_URL}/nomoi/indexes/"
RATE_LIMIT_SECONDS = 1.0  # Delay between requests


def fetch_page(url: str, session: requests.Session, retries: int = 3) -> str | None:
    """Fetch a page with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            # Try to decode with the right encoding
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def discover_index_pages(session: requests.Session) -> list[str]:
    """Phase 1: Get all .html links from the indexes directory."""
    logger.info("Phase 1: Discovering index pages...")
    html = fetch_page(INDEX_DIR_URL, session)
    if not html:
        logger.error("Failed to fetch index directory")
        return []

    soup = BeautifulSoup(html, "html.parser")
    pages = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".html") and not href.startswith(".."):
            full_url = urljoin(INDEX_DIR_URL, href)
            pages.append(full_url)

    logger.info(f"Found {len(pages)} index pages")
    return pages


def extract_pdf_links(index_url: str, session: requests.Session) -> list[str]:
    """Phase 2: Extract PDF links from a single index page."""
    html = fetch_page(index_url, session)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    pdfs = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            full_url = urljoin(index_url, href)
            pdfs.append(full_url)

    return pdfs


def download_file(url: str, output_dir: Path) -> bool:
    """Download a single file, skipping if it already exists. Thread-safe."""
    filename = urlparse(url).path.split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        return True  # Already downloaded

    try:
        time.sleep(0.5)  # Small delay to avoid overwhelming the server
        session = _make_session()
        resp = session.get(url, timeout=60, stream=True)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


# Thread-safe counters for parallel downloads
_download_lock = threading.Lock()
_download_stats = {"downloaded": 0, "failed": 0, "skipped": 0}


def _download_worker(url: str, output_dir: Path) -> bool:
    """Worker for parallel downloads. Updates shared counters."""
    success = download_file(url, output_dir)
    with _download_lock:
        if success:
            _download_stats["downloaded"] += 1
        else:
            _download_stats["failed"] += 1
    return success


def scrape_html_fallback(
    index_pages: list[str],
    pdf_index_pages: set[str],
    output_dir: Path,
    session: requests.Session,
) -> int:
    """Phase 4: For index pages with no PDFs, try to scrape full.html text."""
    count = 0
    for page_url in index_pages:
        if page_url in pdf_index_pages:
            continue

        # Try to construct the full.html URL
        # Index page: /nomoi/indexes/100.html → full text: /nomoi/enop/non-ind/0_100/full.html
        filename = urlparse(page_url).path.split("/")[-1].replace(".html", "")

        # Try numeric Cap laws
        full_url = f"{BASE_URL}/nomoi/enop/non-ind/0_{filename}/full.html"
        html = fetch_page(full_url, session)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        # Extract main content text
        body = soup.find("body")
        if not body:
            continue

        text = body.get_text(separator="\n", strip=True)
        if len(text) < 100:
            continue

        output_path = output_dir / f"law_{filename}.txt"
        output_path.write_text(text, encoding="utf-8")
        count += 1
        logger.info(f"Scraped HTML text: law_{filename}.txt ({len(text)} chars)")

    return count


def main():
    parser = argparse.ArgumentParser(description="Scrape CyLaw.org legal documents")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "test_files_cy"),
        help="Directory to save downloaded files (default: ~/test_files_cy/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "LegalRAG-Research/1.0 (legal document archival)",
    })

    # Cache file for discovered URLs (survives restarts)
    url_cache_path = output_dir / "_pdf_urls_cache.json"

    # Phase 1 & 2: Discover index pages and extract PDF links (or load from cache)
    if url_cache_path.exists():
        logger.info(f"Loading cached PDF URLs from {url_cache_path}")
        cache_data = json.loads(url_cache_path.read_text(encoding="utf-8"))
        all_pdf_urls = set(cache_data["pdf_urls"])
        pdf_index_pages = set(cache_data["pdf_index_pages"])
        index_pages = cache_data["index_pages"]
        logger.info(f"Loaded {len(all_pdf_urls)} PDF URLs from cache (skipping Phase 1 & 2)")
    else:
        index_pages = discover_index_pages(session)
        if not index_pages:
            logger.error("No index pages found. Exiting.")
            return

        # Phase 2: Extract PDF links from all index pages
        logger.info("Phase 2: Extracting PDF links from index pages...")
        all_pdf_urls = set()
        pdf_index_pages = set()  # Track which index pages had PDFs

        for i, page_url in enumerate(index_pages):
            pdfs = extract_pdf_links(page_url, session)
            if pdfs:
                pdf_index_pages.add(page_url)
                all_pdf_urls.update(pdfs)
            if (i + 1) % 50 == 0:
                logger.info(f"  Scanned {i+1}/{len(index_pages)} index pages, found {len(all_pdf_urls)} unique PDFs so far")

        logger.info(f"Found {len(all_pdf_urls)} unique PDF links from {len(pdf_index_pages)} index pages")

        # Save to cache so Phase 2 doesn't repeat on restart
        cache_data = {
            "pdf_urls": sorted(all_pdf_urls),
            "pdf_index_pages": sorted(pdf_index_pages),
            "index_pages": index_pages,
        }
        url_cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved {len(all_pdf_urls)} URLs to cache: {url_cache_path}")

    # Phase 3: Download PDFs (parallel, gentle)
    WORKERS = 5
    pdf_list = sorted(all_pdf_urls)
    logger.info(f"Phase 3: Downloading {len(pdf_list)} PDFs with {WORKERS} parallel workers...")

    # Reset stats
    _download_stats["downloaded"] = 0
    _download_stats["failed"] = 0

    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_download_worker, url, output_dir): url for url in pdf_list}
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                with _download_lock:
                    d, f = _download_stats["downloaded"], _download_stats["failed"]
                logger.info(f"  Progress: {completed}/{len(pdf_list)} ({d} downloaded, {f} failed)")

    downloaded = _download_stats["downloaded"]
    failed = _download_stats["failed"]
    logger.info(f"PDF download complete: {downloaded} downloaded, {failed} failed")

    # Phase 4: HTML fallback
    logger.info("Phase 4: Scraping HTML full text for laws without PDFs...")
    html_count = scrape_html_fallback(index_pages, pdf_index_pages, output_dir, session)
    logger.info(f"HTML scraping complete: {html_count} text files saved")

    # Write manifest
    manifest = {
        "source": "https://www.cylaw.org",
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_pages_scanned": len(index_pages),
        "pdfs_downloaded": downloaded,
        "pdfs_failed": failed,
        "html_texts_scraped": html_count,
        "total_files": downloaded + html_count,
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Manifest written to {manifest_path}")

    # Summary
    total_files = len(list(output_dir.glob("*.pdf"))) + len(list(output_dir.glob("*.txt")))
    logger.info(f"\nDone! {total_files} files in {output_dir}")


if __name__ == "__main__":
    main()
