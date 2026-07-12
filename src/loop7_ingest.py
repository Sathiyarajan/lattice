import os
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader

RAW_DIR = Path(os.path.expanduser("~/projects/lattice/data/raw"))

def parse_txt(path):
    return path.read_text(encoding="utf-8").strip()

def parse_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return soup.get_text(separator=" ", strip=True)

def parse_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

PARSERS = {".txt": parse_txt, ".html": parse_html, ".pdf": parse_pdf}

def ingest(raw_dir):
    results = []
    for path in raw_dir.iterdir():
        parser = PARSERS.get(path.suffix.lower())
        if not parser:
            continue
        text = parser(path)
        results.append({"source": path.name, "text": text})
    return results

docs = ingest(RAW_DIR)
for d in docs:
    print(d["source"], "->", len(d["text"]), "chars")
    print("  ", d["text"][:80])

assert len(docs) >= 2, "expected at least txt and html parsed"
assert all(len(d["text"]) > 0 for d in docs), "empty extraction found"
print("OK")
