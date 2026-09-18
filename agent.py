import json
import hashlib
import time
import yaml
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import CrossEncoder

BASE_DIR = Path(__file__).parent
EXPERIENCE_PATH = BASE_DIR / "experience_library.md"
SOURCES_PATH = BASE_DIR / "sources.yaml"
MEMORY_PATH = BASE_DIR / "memory.json"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
HISTORY_DIR = DOCS_DIR / "history"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Loaders & Helpers
# ---------------------------------------------------------

def load_experience_library():
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return "Mechanical Engineer, P.Eng., PMP, Project Engineering Supervisor, Infrastructure & Facilities."

def load_sources():
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])

def load_memory():
    try:
        if MEMORY_PATH.exists():
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                mem = json.load(f)
                if "cache" not in mem:
                    mem["cache"] = {}
                if "jobs" not in mem:
                    mem["jobs"] = []
                return mem
    except Exception as e:
        print(f"[WARN] Memory file issue: {e} — resetting.")
    return {"jobs": [], "cache": {}}

def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def make_job_id(source_name, title, url):
    raw = f"{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def is_duplicate(job, memory):
    for existing in memory["jobs"]:
        if job["id"] == existing.get("id"):
            return True
        if job["url"] == existing.get("url"):
            return True
    return False

# ---------------------------------------------------------
# Targeted Scraping Engines (Playwright Driven)
# ---------------------------------------------------------

def fetch_page_content(page, url):
    """Navigates to URL with strict 10s timeout to avoid hanging."""
    try:
        # Use domcontentloaded instead of networkidle
        page.goto(url, wait_until="domcontentloaded", timeout=10000)
        time.sleep(1) # Short wait for initial JS
        return page.content()
    except Exception as e:
        print(f"[SKIP] Timeout or error loading {url}")
        return ""

def parse_linkedin(soup, source_name):
    """Targeted parsing for public LinkedIn Search cards."""
    jobs = []
    # Standard LinkedIn Public Card Containers
    cards = soup.find_all("div", class_="base-search-card")
    for card in cards:
        title_tag = card.find("h3", class_="base-search-card__title")
        company_tag = card.find("h4", class_="base-search-card__subtitle")
        link_tag = card.find("a", class_="base-card__full-link")
        time_tag = card.find("time")
        
        if title_tag and link_tag:
            title = title_tag.get_text(strip=True)
            url = link_tag.get("href", "").split("?")[0]
            company = company_tag.get_text(strip=True) if company_tag else "LinkedIn Posting"
            posted = time_tag.get_text(strip=True) if time_tag else "Recent"
            
            jobs.append({
                "id": make_job_id(source_name, title, url),
                "source": source_name,
                "title": title,
                "company": company,
                "url": url,
                "posted": posted
            })
    return jobs

def parse_generic_ats(soup, url, source_name):
    """Fallback parser for Workday, BambooHR, Workable, Deel, etc."""
    jobs = []
    from urllib.parse import urljoin, urlparse
    
    domain_name = urlparse(url).netloc.replace("www.", "").split(".")[0].title()

    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip()
        href = a.get("href") or ""
        
        # Avoid navigation footer links like 'Privacy Policy', 'Cookie', 'Sign In'
        if not text or len(text) < 4 or any(bad in text.lower() for bad in ["privacy", "terms", "sign in", "cookie", "login", "easy apply"]):
            continue
            
        # Match URL patterns indicative of actual job postings
        if any(keyword in href.lower() for keyword in ["/job/", "/careers/", "/o/", "/jobs/"]) or \
           any(keyword in text.lower() for keyword in ["engineer", "manager", "lead", "director", "specialist"]):
            
            full_url = urljoin(url, href)
            job_id = make_job_id(source_name, text, full_url)
            
            jobs.append({
                "id": job_id,
                "source": source_name,
                "title": text,
                "company": domain_name,
                "url": full_url,
                "posted": "Recent"
            })
    return jobs

def extract_detailed_job_text(page, url):
    """Fast extraction for individual job descriptions."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=8000)
        soup = BeautifulSoup(page.content(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:2500]
    except Exception:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")
    # Clean noise tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
        
    text = soup.get_text(separator=" ", strip=True)
    return text[:3000] # Limit size for cross-encoder performance

# ---------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------

def write_html_report(scored_jobs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    report_name = f"report-{today}.html"
    report_path = REPORTS_DIR / report_name

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Job Matching Report — {today}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
  h1 {{ margin-bottom: 5px; color: #1a252f; }}
  .sub {{ color: #6c757d; margin-bottom: 20px; }}
  #searchBox {{ padding: 10px; width: 320px; font-size: 14px; margin-bottom: 20px; border: 1px solid #ced4da; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
  th {{ background-color: #0066cc; color: white; cursor: pointer; }}
  tr:hover {{ background-color: #f1f5f9; }}
  .score-badge {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; background: #e3f2fd; color: #0d47a1; }}
  a.btn {{ text-decoration: none; background: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 13px; }}
</style>
<script>
function filterTable() {{
  var input = document.getElementById("searchBox").value.toLowerCase();
  var rows = document.getElementById("jobTable").rows;
  for (var i = 1; i < rows.length; i++) {{
    var rowText = rows[i].innerText.toLowerCase();
    rows[i].style.display = rowText.includes(input) ? "" : "none";
  }}
}}
</script>
</head>
<body>

<h1>🎯 Job Agent Daily Recommendations</h1>
<p class="sub">Generated on {today} using Cross-Encoder AI Semantic Matching</p>
<input type="text" id="searchBox" onkeyup="filterTable()" placeholder="Search title, company, or source...">

<table id="jobTable">
<thead>
<tr>
  <th>Score</th>
  <th>Job Title</th>
  <th>Company</th>
  <th>Source</th>
  <th>Posted</th>
  <th>Action</th>
</tr>
</thead>
<tbody>
"""
    for job in scored_jobs:
        html += f"""
<tr>
  <td><span class="score-badge">{job['score']:.2f}</span></td>
  <td><strong>{job['title']}</strong></td>
  <td>{job['company']}</td>
  <td>{job['source']}</td>
  <td>{job['posted']}</td>
  <td><a href="{job['url']}" target="_blank" class="btn">View Posting</a></td>
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

    # Save to current.html for easy GitHub Pages viewing
    with open(DOCS_DIR / "current.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------

def main():
    print("[INFO] Initializing Cross-Encoder Model for Job Matching...")
    # MS-MARCO Cross-Encoder calculates precise relevance scores between (Resume, Job Text)
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    experience_text = load_experience_library()[:1500] # Use core executive profile summary
    sources = load_sources()
    memory = load_memory()
    
    discovered_jobs = []

    with sync_playwright() as p:
        # Launch Chromium with anti-detection headers
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for src in sources:
            name = src["name"]
            url = src["url"]
            print(f"[INFO] Fetching listings from {name} -> {url}")

            html = fetch_page_content(page, url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Route parsing depending on site source
            if "linkedin" in name.lower():
                jobs = parse_linkedin(soup, name)
            else:
                jobs = parse_generic_ats(soup, url, name)

            print(f"[INFO] Found {len(jobs)} candidates on {name}")

            for job in jobs:
                if not is_duplicate(job, memory):
                    # Fetch detailed job text for accurate matching
                    body_text = extract_detailed_job_text(page, job["url"])
                    job["body_text"] = body_text if body_text else job["title"]
                    discovered_jobs.append(job)

        browser.close()

    print(f"[INFO] Processing cross-encoder matching scores for {len(discovered_jobs)} new listings...")
    
    if discovered_jobs:
        # Prepare pairs for Cross-Encoder scoring: (Resume Summary, Job Title + Body)
        pairs = [[experience_text, f"{j['title']} at {j['company']}: {j['body_text']}"] for j in discovered_jobs]
        scores = model.predict(pairs)

        for job, score in zip(discovered_jobs, scores):
            job["score"] = float(score)
            memory["jobs"].append(job)

        # Sort jobs highest to lowest match score
        discovered_jobs.sort(key=lambda x: x["score"], reverse=True)
        
    save_memory(memory)
    write_html_report(discovered_jobs)
    print(f"[INFO] Workflow complete. Report generated with {len(discovered_jobs)} scored jobs.")

if __name__ == "__main__":
    main()