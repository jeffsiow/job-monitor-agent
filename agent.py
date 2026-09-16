import json
import hashlib
import requests
import yaml
import os
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer

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
            mem = json.load(f)
            if "cache" not in mem:
                mem["cache"] = {}
            return mem
    return {"jobs": [], "cache": {}}

def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# ---------- Duplicate suppression ----------
def is_duplicate(job, memory):
    for existing in memory["jobs"]:
        if job["id"] == existing["id"]:
            return True
        if job["url"] == existing["url"]:
            return True
        if job["title"].lower() == existing["title"].lower():
            return True
    return False

# ---------- Job ID ----------
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

    return "\n".join(texts)[:5000]

# ---------- Cosine similarity ----------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)

# ---------- HTML Report ----------
def write_html_report(scored_jobs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"report-{today}.html"

    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Job Agent Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 8px 12px; border: 1px solid #ccc; }
th { cursor: pointer; background: #f2f2f2; }
tr:hover { background: #fafafa; }
#searchBox { margin-bottom: 12px; padding: 8px; width: 300px; }
</style>
<script>
function sortTable(n) {
  var table = document.getElementById("jobTable");
  var rows = table.rows;
  var switching = true;
  var dir = "desc";
  while (switching) {
    switching = false;
    for (var i = 1; i < rows.length - 1; i++) {
      var x = rows[i].getElementsByTagName("TD")[n];
      var y = rows[i + 1].getElementsByTagName("TD")[n];
      var cmp = (n === 2) ? parseFloat(x.innerHTML) - parseFloat(y.innerHTML)
                          : x.innerHTML.localeCompare(y.innerHTML);
      if ((dir === "asc" && cmp > 0) || (dir === "desc" && cmp < 0)) {
        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
        switching = true;
        break;
      }
    }
    if (!switching && dir === "desc") {
      dir = "asc";
      switching = true;
    }
  }
}

function filterTable() {
  var input = document.getElementById("searchBox").value.toLowerCase();
  var rows = document.getElementById("jobTable").rows;
  for (var i = 1; i < rows.length; i++) {
    var rowText = rows[i].innerText.toLowerCase();
    rows[i].style.display = rowText.includes(input) ? "" : "none";
  }
}
</script>
</head>
<body>

<h1>Job Agent Report — """ + today + """</h1>
<input type="text" id="searchBox" onkeyup="filterTable()" placeholder="Search jobs...">

<table id="jobTable">
<thead>
<tr>
<th onclick="sortTable(0)">Title</th>
<th onclick="sortTable(1)">Source</th>
<th onclick="sortTable(2)">Score</th>
<th>Link</th>
<th>Snippet</th>
</tr>
</thead>
<tbody>
"""

    for job in scored_jobs:
        html += f"""
<tr>
<td>{job['title']}</td>
<td>{job['source']}</td>
<td>{job['score']:.4f}</td>
<td><a href="{job['url']}" target="_blank">Open</a></td>
<td>{job['snippet']}</td>
</tr>
"""

    html += """
</tbody>
</table>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Saved HTML report → {report_path}")

# ---------- Main pipeline ----------
def main():
    # Load free embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    experience_text = load_experience_library()
    experience_embedding = model.encode(experience_text)

    sources = load_sources()
    memory = load_memory()
    cache = memory["cache"]

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

        for job in jobs:
            if is_duplicate(job, memory):
                continue

            url = job["url"]

            # ---------- CACHE CHECK ----------
            if url in cache:
                print(f"[CACHE] Using cached embedding for {url}")
                job_embedding = cache[url]["embedding"]
                snippet = cache[url]["snippet"]
            else:
                job_html = fetch_html(url)
                job_text = extract_job_text(job_html)

                if not job_text:
                    continue

                job_embedding = model.encode(job_text)
                snippet = job_text[:300]

                cache[url] = {
                    "embedding": job_embedding,
                    "snippet": snippet
                }

            score = cosine_similarity(experience_embedding, job_embedding)

            job["score"] = score
            job["snippet"] = snippet

            scored_jobs.append(job)
            memory["jobs"].append(job)

    save_memory(memory)

    scored_jobs.sort(key=lambda x: x["score"], reverse=True)

    write_html_report(scored_jobs)

if __name__ == "__main__":
    main()
