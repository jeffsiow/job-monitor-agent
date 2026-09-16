import json
import hashlib
import requests
import yaml
import os
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from openai import OpenAI

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
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text
    except:
        return ""

# ---------- Extract job description ----------
def extract_job_text(html):
    soup = BeautifulSoup(html, "html.parser")
    texts = []

    for tag in soup.find_all(["p", "li", "div"]):
        t = tag.get_text(" ", strip=True)
        if t:
            texts.append(t)

    return "\n".join(texts)[:5000]  # safety limit

# ---------- Embeddings ----------
def embed_text(client, text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None

# ---------- Cosine similarity ----------
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------- Main pipeline ----------
def main():
    # OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load experience library and embed it
    experience_text = load_experience_library()
    experience_embedding = embed_text(client, experience_text)

    sources = load_sources()
    memory = load_memory()
    seen_ids = {job["id"] for job in memory["jobs"]}

    scored_jobs = []

    for src in sources:
        name = src["name"]
        url = src["url"]
        print(f"[INFO] Fetching {name} -> {url}")

        html = fetch_html(url)
        if not html:
            print(f"[WARN] No HTML returned for {name}")
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Find job links
        jobs = []
        for a in soup.find_all("a"):
            text = (a.get_text() or "").strip()
            href = a.get("href") or ""
            if not text or not href:
                continue

            if "job" in href.lower() or "careers" in href.lower() or "jobs" in text.lower():
                from urllib.parse import urljoin
                full_url = urljoin(url, href) if href.startswith("/") else href

                job_id = make_job_id(name, text, full_url)
                jobs.append({
                    "id": job_id,
                    "source": name,
                    "title": text,
                    "url": full_url
                })

        print(f"[INFO] Found {len(jobs)} candidate links on {name}")

        # Process each job
        for job in jobs:
            if job["id"] in seen_ids:
                continue

            job_html = fetch_html(job["url"])
            job_text = extract_job_text(job_html)

            if not job_text:
                continue

            job_embedding = embed_text(client, job_text)
            if not job_embedding:
                continue

            score = cosine_similarity(experience_embedding, job_embedding)

            job["score"] = score
            job["snippet"] = job_text[:300]

            scored_jobs.append(job)
            memory["jobs"].append(job)
            seen_ids.add(job["id"])

    save_memory(memory)

    # Sort by score
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)

    # Write Markdown report
    today = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"report-{today}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Job Agent Report — {today}\n\n")
        f.write("Ranked job matches based on your master experience library.\n\n")

        for job in scored_jobs:
            f.write(f"## {job['title']}\n")
            f.write(f"**Source:** {job['source']}\n\n")
            f.write(f"**Score:** {job['score']:.4f}\n\n")
            f.write(f"[Job Link]({job['url']})\n\n")
            f.write(f"**Snippet:**\n\n{job['snippet']}\n\n")
            f.write("---\n\n")

    print(f"[INFO] Saved ranked report with {len(scored_jobs)} jobs → {report_path}")

if __name__ == "__main__":
    main()
