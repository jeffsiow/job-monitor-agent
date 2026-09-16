import json
import hashlib
import requests
import yaml
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
EXPERIENCE_PATH = BASE_DIR / "experience_library.md"
SOURCES_PATH = BASE_DIR / "sources.yaml"
MEMORY_PATH = BASE_DIR / "memory.json"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ---------- Load experience library ----------
def load_experience_library():
    return EXPERIENCE_PATH.read_text(encoding="utf-8")

# ---------- Load sources ----------
def load_sources():
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["sources"]

# ---------- Load memory ----------
def load_memory():
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"jobs": []}

def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# ---------- Simple job model ----------
def make_job_id(source_name, title, url):
    raw = f"{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# ---------- HTML fetch ----------
def fetch_html(url):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text

# ---------- Parsers (placeholder for now) ----------
def parse_jobs_from_html(source_name, url, html):
    soup = BeautifulSoup(html, "html.parser")
    jobs = []

    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip()
        href = a.get("href") or ""
        if not text or not href:
            continue

        if "job" in href.lower() or "careers" in href.lower() or "jobs" in text.lower():
            from urllib.parse import urljoin
            full_url = urljoin(url, href) if href.startswith("/") else href

            job_id = make_job_id(source_name, text, full_url)
            jobs.append({
                "id": job_id,
                "source": source_name,
                "title": text,
                "url": full_url
            })

    return jobs

# ---------- Main pipeline ----------
def main():
    experience_text = load_experience_library()
    sources = load_sources()
    memory = load_memory()

    seen_ids = {job["id"] for job in memory["jobs"]}
    new_jobs = []

    for src in sources:
        name = src["name"]
        url = src["url"]
        print(f"[INFO] Fetching {name} -> {url}")

        try:
            html = fetch_html(url)
            jobs = parse_jobs_from_html(name, url, html)
            print(f"[INFO] Found {len(jobs)} candidate links on {name}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch/parse {name}: {e}")
            continue

        for job in jobs:
            if job["id"] not in seen_ids:
                new_jobs.append(job)
                memory["jobs"].append(job)
                seen_ids.add(job["id"])

    save_memory(memory)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"report-{today}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"new_jobs": new_jobs}, f, indent=2)

    print(f"[INFO] Saved report with {len(new_jobs)} new jobs to {report_path}")

if __name__ == "__main__":
    main()
