import os
import json
import hashlib
import time
import requests
import yaml
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import CrossEncoder

BASE_DIR = Path(__file__).parent
EXPERIENCE_PATH = BASE_DIR / "experience_library.md"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
CACHE_PATH = BASE_DIR / "jobs_cache.json"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

def load_experience_library():
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return "Mechanical Engineer, P.Eng., PMP, Project Manager, CleanTech, Energy."

def make_job_id(source_name, title, url):
    raw = f"{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read jobs_cache.json: {e}")
    return {}

def save_cache(cache_data):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to write jobs_cache.json: {e}")

# ---------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------

def fetch_google_jobs(query, location="Calgary, AB"):
    """Fetches Google Jobs via SerpAPI filtered for jobs posted in the past week."""
    jobs = []
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("[WARN] SERPAPI_KEY not found in environment. Skipping Google Jobs.")
        return jobs

    url = "https://serpapi.com/search.json"
    for start in [0, 10]:
        params = {
            "engine": "google_jobs",
            "q": f"{query} {location}",
            "chips": "date_posted:week",  # Filter for postings in the last week
            "start": start,
            "api_key": api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                results = resp.json().get("jobs_results", [])
                for item in results:
                    title = item.get("title", "")
                    company = item.get("company_name", "")
                    link = item.get("share_link") or (item.get("related_links", [{}])[0].get("link", ""))
                    desc = item.get("description", "")
                    
                    if title and link:
                        jobs.append({
                            "id": make_job_id("google_jobs", title, link),
                            "source": "google_jobs",
                            "title": title,
                            "company": company,
                            "url": link,
                            "body_text": f"{title} at {company}: {desc[:1500]}",
                            "posted": item.get("detected_extensions", {}).get("posted_at", "Past week")
                        })
        except Exception as e:
            print(f"[WARN] Google Jobs batch failed for '{query}' (start={start}): {e}")
            
    print(f"[INFO] Google Jobs ('{query}'): Found {len(jobs)} postings.")
    return jobs

def fetch_bamboohr_eavor():
    jobs = []
    url = "https://eavortechnologies.bamboohr.com/careers/list"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        if resp.status_code == 200:
            for result in resp.json().get("result", []):
                title = result.get("jobOpeningName", "")
                job_id_num = result.get("id", "")
                link = f"https://eavortechnologies.bamboohr.com/careers/{job_id_num}"
                jobs.append({
                    "id": make_job_id("eavor", title, link),
                    "source": "eavor",
                    "title": title,
                    "company": "Eavor Technologies",
                    "url": link,
                    "body_text": f"{title} - Eavor Technologies Calgary",
                    "posted": "Recent"
                })
            print(f"[INFO] Eavor: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Eavor API failed: {e}")
    return jobs

def fetch_workable_seeq():
    jobs = []
    url = "https://apply.workable.com/api/v3/accounts/seeq/jobs"
    try:
        resp = requests.post(url, json={"query": "", "location": [], "department": []}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = item.get("title", "")
                shortcode = item.get("shortcode", "")
                link = f"https://apply.workable.com/seeq/j/{shortcode}/"
                jobs.append({
                    "id": make_job_id("seeq", title, link),
                    "source": "seeq",
                    "title": title,
                    "company": "Seeq",
                    "url": link,
                    "body_text": f"{title} - Seeq Careers",
                    "posted": "Recent"
                })
            print(f"[INFO] Seeq: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Seeq API failed: {e}")
    return jobs

def fetch_black_veatch_playwright():
    """Renders Black & Veatch career site querying both targeted search URLs."""
    jobs = []
    urls = [
        "https://careers.bv.com/search/?createNewAlert=false&q=project+manager&locationsearch=canada+OR+united+states&optionsFacetsDD_customfield3=&optionsFacetsDD_customfield5=Project+Management",
        "https://careers.bv.com/search/?createNewAlert=false&q=project+manager&locationsearch=canada&optionsFacetsDD_customfield3=&optionsFacetsDD_customfield5="
    ]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            for url in urls:
                page.goto(url, wait_until="domcontentloaded", timeout=35000)
                time.sleep(4)
                
                soup = BeautifulSoup(page.content(), "html.parser")
                for a in soup.find_all("a", href=True):
                    title = a.get_text(strip=True)
                    href = a["href"]
                    if "/job/" in href.lower() and len(title) > 3 and "search" not in title.lower():
                        full_url = href if href.startswith("http") else f"https://careers.bv.com{href}"
                        jobs.append({
                            "id": make_job_id("black_veatch", title, full_url),
                            "source": "black_veatch",
                            "title": title,
                            "company": "Black & Veatch",
                            "url": full_url,
                            "body_text": f"{title} - Black & Veatch Project Management",
                            "posted": "Recent"
                        })
            browser.close()
            print(f"[INFO] Black & Veatch (Playwright): Found {len(jobs)} total postings across both URLs.")
    except Exception as e:
        print(f"[WARN] Black & Veatch Playwright fetch failed: {e}")
    return jobs

def fetch_kanin_energy():
    jobs = []
    url = "https://kaninenergy.com/wp-json/wp/v2/pages?slug=careers"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.json()) > 0:
            content = resp.json()[0].get("content", {}).get("rendered", "")
            soup = BeautifulSoup(content, "html.parser")
            elements = soup.find_all(["h2", "h3", "h4", "p", "a"])
            for el in elements:
                text = el.get_text(strip=True)
                if any(k in text.lower() for k in ["engineer", "manager", "director", "lead", "developer", "specialist"]):
                    link = el.get("href") if el.name == "a" else "https://kaninenergy.com/careers/"
                    jobs.append({
                        "id": make_job_id("kanin_energy", text, link),
                        "source": "kanin_energy",
                        "title": text,
                        "company": "Kanin Energy",
                        "url": link,
                        "body_text": f"{text} at Kanin Energy",
                        "posted": "Recent"
                    })
            print(f"[INFO] Kanin Energy: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Kanin Energy fetch failed: {e}")
    return jobs

def fetch_city_of_calgary_playwright():
    """Extracts job titles directly from PeopleSoft iframe or root page."""
    jobs = []
    url = "https://recruiting.calgary.ca/psc/hcm/EMPLOYEE/HRMS/c/HRS_HRAM_FL.HRS_CG_SEARCH_FL.GBL?Page=HRS_APP_SCHJOB_FL&Action=U"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page.goto(url, wait_until="networkidle", timeout=40000)
            time.sleep(5)
            
            frames = page.frames
            for frame in frames:
                try:
                    content = frame.content()
                    soup = BeautifulSoup(content, "html.parser")
                    for a in soup.find_all(["a", "span"]):
                        title = a.get_text(strip=True)
                        if any(k in title.lower() for k in ["engineer", "manager", "planner", "analyst", "lead", "coordinator", "officer", "specialist", "project"]):
                            jobs.append({
                                "id": make_job_id("city_of_calgary", title, url),
                                "source": "city_of_calgary",
                                "title": title,
                                "company": "City of Calgary",
                                "url": url,
                                "body_text": f"{title} - City of Calgary Careers",
                                "posted": "Recent"
                            })
                except Exception:
                    continue

            browser.close()
            print(f"[INFO] City of Calgary (Playwright): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] City of Calgary Playwright fetch failed: {e}")
    return jobs

def fetch_climate_tech_list_playwright():
    """Renders Climate Tech List using Playwright to extract dynamically rendered / Airtable job cards."""
    jobs = []
    url = "https://www.climatetechlist.com/jobs?location=calgary"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page.goto(url, wait_until="networkidle", timeout=40000)
            time.sleep(6)

            sources_to_check = [page] + page.frames
            for src in sources_to_check:
                try:
                    soup = BeautifulSoup(src.content(), "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        title = a.get_text(strip=True)
                        if ("/job/" in href.lower() or "/posting" in href.lower() or "airtable.com" in href.lower()) and len(title) > 3:
                            full_url = href if href.startswith("http") else f"https://www.climatetechlist.com{href}"
                            jobs.append({
                                "id": make_job_id("climate_tech_list", title, full_url),
                                "source": "climate_tech_list",
                                "title": title,
                                "company": "Climate Tech List",
                                "url": full_url,
                                "body_text": f"{title} - Climate Tech List Calgary",
                                "posted": "Recent"
                            })
                except Exception:
                    continue

            browser.close()
            print(f"[INFO] Climate Tech List (Playwright): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Climate Tech List Playwright fetch failed: {e}")
    return jobs

# ---------------------------------------------------------
# Report Writer
# ---------------------------------------------------------

def write_html_report(scored_jobs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Job Matching Report — {today}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
  h1 {{ margin-bottom: 5px; color: #1a252f; }}
  .sub {{ color: #6c757d; margin-bottom: 20px; }}
  #searchBox {{ padding: 10px; width: 320px; font-size: 14px; margin-bottom: 20px; border: 1px solid #ced4da; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
  th {{ background-color: #0066cc; color: white; cursor: pointer; user-select: none; }}
  th:hover {{ background-color: #0052a5; }}
  th.asc::after {{ content: " ▲"; font-size: 10px; }}
  th.desc::after {{ content: " ▼"; font-size: 10px; }}
  tr:hover {{ background-color: #f1f5f9; }}
  .score-badge {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; background: #e3f2fd; color: #0d47a1; }}
  a.btn {{ text-decoration: none; background: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 13px; display: inline-block; }}
  a.btn:hover {{ background: #218838; }}
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

function sortTable(columnIndex) {{
  var table = document.getElementById("jobTable");
  var rows = Array.from(table.rows).slice(1);
  var headers = table.querySelectorAll("th");
  var currentHeader = headers[columnIndex];
  var isAscending = currentHeader.classList.contains("asc");

  headers.forEach(h => h.classList.remove("asc", "desc"));

  rows.sort(function(rowA, rowB) {{
    var cellA = rowA.cells[columnIndex].innerText.trim();
    var cellB = rowB.cells[columnIndex].innerText.trim();

    var numA = parseFloat(cellA);
    var numB = parseFloat(cellB);

    if (!isNaN(numA) && !isNaN(numB)) {{
      return isAscending ? numA - numB : numB - numA;
    }}

    return isAscending 
      ? cellA.localeCompare(cellB, undefined, {{numeric: true, sensitivity: 'base'}})
      : cellB.localeCompare(cellA, undefined, {{numeric: true, sensitivity: 'base'}});
  }});

  currentHeader.classList.add(isAscending ? "desc" : "asc");

  var tbody = table.querySelector("tbody");
  rows.forEach(row => tbody.appendChild(row));
}}
</script>
</head>
<body>

<h1>🎯 Daily Job Match Recommendations</h1>
<p class="sub">Generated on {today} • Click column headers to sort</p>
<input type="text" id="searchBox" onkeyup="filterTable()" placeholder="Filter by title, company, or source...">

<table id="jobTable">
<thead>
<tr>
  <th onclick="sortTable(0)" class="desc">Score</th>
  <th onclick="sortTable(1)">Job Title</th>
  <th onclick="sortTable(2)">Company</th>
  <th onclick="sortTable(3)">Date Posted</th>
  <th onclick="sortTable(4)">Source</th>
  <th onclick="sortTable(5)">Action</th>
</tr>
</thead>
<tbody>
"""
    for job in scored_jobs:
        posted_date = job.get('posted') or job.get('first_seen', 'Recent')
        html += f"""
<tr>
  <td><span class="score-badge">{job.get('score', 0.0):.2f}</span></td>
  <td><strong>{job['title']}</strong></td>
  <td>{job['company']}</td>
  <td>{posted_date}</td>
  <td>{job['source']}</td>
  <td><a href="{job['url']}" target="_blank" class="btn">View Posting</a></td>
</tr>
"""
    html += """
</tbody>
</table>
</body>
</html>
"""

    with open(REPORTS_DIR / f"report-{today}.html", "w", encoding="utf-8") as f:
        f.write(html)
    with open(DOCS_DIR / "current.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------------------------------------
# Pipeline Entry Point
# ---------------------------------------------------------

def main():
    print("[INFO] Loading local cache...")
    cache = load_cache()

    print("[INFO] Initializing Cross-Encoder Model...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    experience_text = load_experience_library()[:1500]
    
    discovered_jobs = []

    # 1. Google Jobs (Weekly filter)
    print("[INFO] Fetching Google Jobs...")
    google_queries = [
        "Project Engineer",
        "Project Manager",
        "Project Director",
        "Operations Engineer"
    ]
    for q in google_queries:
        discovered_jobs.extend(fetch_google_jobs(q, "Calgary, AB"))

    # 2. Corporate APIs & Direct Fetchers
    print("[INFO] Fetching Eavor...")
    discovered_jobs.extend(fetch_bamboohr_eavor())

    print("[INFO] Fetching Seeq...")
    discovered_jobs.extend(fetch_workable_seeq())

    print("[INFO] Fetching Black & Veatch...")
    discovered_jobs.extend(fetch_black_veatch_playwright())

    print("[INFO] Fetching Kanin Energy...")
    discovered_jobs.extend(fetch_kanin_energy())

    # 3. Playwright Fetchers
    print("[INFO] Fetching Climate Tech List...")
    discovered_jobs.extend(fetch_climate_tech_list_playwright())

    print("[INFO] Fetching City of Calgary...")
    discovered_jobs.extend(fetch_city_of_calgary_playwright())

    # Update cache with newly discovered items
    new_jobs_count = 0
    for j in discovered_jobs:
        job_id = j["id"]
        if job_id not in cache:
            j["first_seen"] = datetime.utcnow().strftime("%Y-%m-%d")
            cache[job_id] = j
            new_jobs_count += 1
        else:
            cache[job_id].update(j)

    print(f"[INFO] Added {new_jobs_count} new postings to local cache. Total cached jobs: {len(cache)}")
    save_cache(cache)

    all_jobs = list(cache.values())

    if all_jobs:
        pairs = [[experience_text, f"{j['title']} at {j['company']}: {j['body_text']}"] for j in all_jobs]
        scores = model.predict(pairs)

        for job, score in zip(all_jobs, scores):
            job["score"] = float(score)

        all_jobs.sort(key=lambda x: x["score"], reverse=True)

    write_html_report(all_jobs)
    print(f"[INFO] Complete! Output saved to docs/current.html with {len(all_jobs)} total listings.")

if __name__ == "__main__":
    main()
