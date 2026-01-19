import os
import requests
import zipfile
import tempfile
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from storage.tender_store import upsert_tender
from storage.pdf_store import upsert_pdf_metadata

BASE_URL = "https://www.mha.gov.in/en/tenders"
BASE_DOMAIN = "https://www.mha.gov.in"

PDF_DIR = "data/pdfs/MHA"
ZIP_DIR = "data/zips/MHA"
ZIP_PATH = os.path.join(ZIP_DIR, "mha_all_tenders_pdfs.zip")

HEADERS = {
    "User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"
}

DOWNLOAD_TIMEOUT = (5, 30)  # (connect, read) seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

def download_pdf(pdf_url, pdf_path, headers):
    if os.path.exists(pdf_path):
        return True

    tmp_dir = os.path.dirname(pdf_path)
    os.makedirs(tmp_dir, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                pdf_url,
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT,
                stream=True
            ) as r:
                r.raise_for_status()
                content_type = (r.headers.get("Content-Type") or "").lower()
                if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                    return False

                fd, tmp_path = tempfile.mkstemp(
                    prefix="tmp_",
                    suffix=".pdf",
                    dir=tmp_dir
                )
                try:
                    with os.fdopen(fd, "wb") as f:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)
                    os.replace(tmp_path, pdf_path)
                    return True
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        except Exception:
            if attempt == MAX_RETRIES:
                return False
            time.sleep(RETRY_BACKOFF ** attempt)

    return False

def fetch_mha_tenders():
    print(" Fetching MHA tenders (Block-1)...")

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)

    page = 0
    pdf_files = []

    while True:
        print(f"\n Page {page}")
        page_url = f"{BASE_URL}?page={page}"

        try:
            res = requests.get(page_url, headers=HEADERS, timeout=30)
            res.raise_for_status()
        except Exception:
            break

        try:
            soup = BeautifulSoup(res.text, "lxml")
        except Exception:
            soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table")
        if not table:
            break

        tbody = table.find("tbody")
        if not tbody:
            break

        rows = tbody.find_all("tr")
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            sr_no = cols[0].get_text(strip=True)
            tender_no = cols[1].get_text(strip=True)
            title = cols[2].get_text(strip=True)
            duration = cols[4].get_text(strip=True)

            pdf_tag = cols[3].find("a", href=True)
            if not pdf_tag:
                continue

            pdf_url = urljoin(BASE_DOMAIN, pdf_tag["href"])
            pdf_name = pdf_url.split("/")[-1]
            pdf_path = os.path.join(PDF_DIR, pdf_name)

            tender_id = upsert_tender({
                "source": "MHA",
                "tender_ref_no": tender_no,
                "sr_no": sr_no,
                "title": title,
                "duration": duration,
                "page_no": page
            })

            if not download_pdf(pdf_url, pdf_path, HEADERS):
                continue

            if not os.path.exists(pdf_path):
                continue

            pdf_files.append(pdf_path)

            upsert_pdf_metadata({
                "tender_id": tender_id,
                "tender_ref_no": tender_no,
                "source": "MHA",
                "document_name": pdf_name,
                "document_type": "MHA_PDF",
                "local_path": pdf_path,
                "pdf_url": pdf_url,
                "size_kb": round(os.path.getsize(pdf_path) / 1024, 2),
                "docling_status": "pending"
            })

        page += 1

    if pdf_files:
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in set(pdf_files):
                zipf.write(f, arcname=os.path.basename(f))

        print(f"📦 ZIP created with {len(set(pdf_files))} PDFs")
