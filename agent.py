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

def compute_posting_age(first_seen_str):
    """Calculates age in days based on when the job was first saved in the cache."""
    if not first_seen_str:
        return "New"
    try:
        first_seen_dt = datetime.strptime(first_seen_str, "%Y-%m-%d")
        delta = (datetime.utcnow() - first_seen_dt).days
        if delta == 0:
            return "Today"
        elif delta == 1:
            return "1 day ago"
        else:
            return f"{delta} days ago"
    except Exception:
        return first_seen_str

# ---------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------

def fetch_google_jobs(query, location="Calgary, AB"):
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
            "chips": "date_posted:week",
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
                    "body_text": f"{title} - Eavor Technologies Calgary"
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
                    "body_text": f"{title} - Seeq Careers"
                })
            print(f"[INFO] Seeq: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Seeq API failed: {e}")
    return jobs

def fetch_black_veatch_playwright():
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
                            "body_text": f"{title} - Black & Veatch Project Management"
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
                        "body_text": f"{text} at Kanin Energy"
                    })
            print(f"[INFO] Kanin Energy: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Kanin Energy fetch failed: {e}")
    return jobs

def fetch_city_of_calgary_playwright():
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
                                "body_text": f"{title} - City of Calgary Careers"
                            })
                except Exception:
                    continue

            browser.close()
            print(f"[INFO] City of Calgary (Playwright): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] City of Calgary Playwright fetch failed: {e}")
    return jobs

def fetch_climate_tech_list_playwright():
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
                                "body_text": f"{title} - Climate Tech List Calgary"
                            })
                except Exception:
                    continue

            browser.close()
            print(f"[INFO] Climate Tech List (Playwright): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Climate Tech List Playwright fetch failed: {e}")
    return jobs

# ---------------------------------------------------------
# Phase 2 Interactive Report Writer
# ---------------------------------------------------------

def write_html_report(scored_jobs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Job Matching Dashboard — {today}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
  h1 {{ margin-bottom: 5px; color: #1a252f; }}
  .sub {{ color: #6c757d; margin-bottom: 20px; }}
  
  .token-bar {{ background: #fff3cd; border: 1px solid #ffeeba; padding: 12px 15px; border-radius: 6px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
  .token-bar input {{ flex: 1; padding: 6px 10px; border: 1px solid #ced4da; border-radius: 4px; font-size: 13px; }}
  .token-bar button {{ padding: 6px 12px; background: #856404; color: white; border: none; border-radius: 4px; cursor: pointer; }}

  .tabs {{ display: flex; gap: 8px; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }}
  .tab {{ padding: 8px 16px; border: none; background: #e9ecef; border-radius: 4px; cursor: pointer; font-weight: 600; color: #495057; }}
  .tab.active {{ background: #0066cc; color: white; }}
  
  .controls {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
  #searchBox {{ padding: 9px 12px; width: 320px; font-size: 14px; border: 1px solid #ced4da; border-radius: 4px; }}

  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
  th {{ background-color: #0066cc; color: white; cursor: pointer; user-select: none; }}
  th:hover {{ background-color: #0052a5; }}
  th.asc::after {{ content: " ▲"; font-size: 10px; }}
  th.desc::after {{ content: " ▼"; font-size: 10px; }}
  tr:hover {{ background-color: #f1f5f9; }}

  .score-badge {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; background: #e3f2fd; color: #0d47a1; }}
  
  /* Status Buttons */
  .btn-group {{ display: flex; gap: 4px; }}
  .btn-action {{ border: 1px solid #ced4da; background: white; padding: 5px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; transition: all 0.2s; }}
  .btn-action:hover {{ background: #e2e8f0; }}
  .btn-action.active-new {{ background: #e2e8f0; font-weight: bold; }}
  .btn-action.active-interested {{ background: #fff3cd; border-color: #ffeeba; color: #856404; font-weight: bold; }}
  .btn-action.active-applied {{ background: #d4edda; border-color: #c3e6cb; color: #155724; font-weight: bold; }}
  .btn-action.active-dismissed {{ background: #f8d7da; border-color: #f5c6cb; color: #721c24; font-weight: bold; }}

  a.btn-link {{ text-decoration: none; background: #0066cc; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; display: inline-block; }}
  a.btn-link:hover {{ background: #0052a5; }}
  
  .status-tag {{ font-size: 11px; padding: 2px 6px; border-radius: 3px; text-transform: uppercase; font-weight: bold; display: inline-block; margin-bottom: 4px; }}
  .status-new {{ background: #e2e8f0; color: #475569; }}
  .status-interested {{ background: #fef08a; color: #854d0e; }}
  .status-applied {{ background: #bbf7d0; color: #166534; }}
  .status-dismissed {{ background: #fecdd3; color: #9f1239; }}
</style>
<script>
let currentTab = 'new';

function setGithubToken() {{
  const token = document.getElementById('ghTokenInput').value.trim();
  if (token) {{
    localStorage.setItem('gh_pat', token);
    alert('GitHub Token saved locally in browser!');
  }} else {{
    localStorage.removeItem('gh_pat');
    alert('Token cleared.');
  }}
}}

window.onload = function() {{
  const savedToken = localStorage.getItem('gh_pat');
  if (savedToken) {{
    document.getElementById('ghTokenInput').value = savedToken;
  }}
  filterTab('new');
}};

function filterTab(tabName) {{
  currentTab = tabName;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + tabName).classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const searchInput = document.getElementById("searchBox").value.toLowerCase();
  const rows = document.querySelectorAll("#jobTable tbody tr");

  rows.forEach(row => {{
    const rowStatus = row.getAttribute("data-status") || "new";
    const rowText = row.innerText.toLowerCase();

    const matchesTab = (currentTab === 'all') || (rowStatus === currentTab);
    const matchesSearch = rowText.includes(searchInput);

    row.style.display = (matchesTab && matchesSearch) ? "" : "none";
  }});
}}

async function updateJobStatus(jobId, newStatus) {{
  const token = localStorage.getItem('gh_pat');
  if (!token) {{
    alert("Please enter a GitHub Personal Access Token (with repo write permissions) above to enable 1-click status updating.");
    return;
  }}

  // Identify repo dynamically from location or prompt user once
  let repoPath = localStorage.getItem('gh_repo');
  if (!repoPath) {{
    repoPath = prompt("Enter your GitHub repository path (e.g. username/job-matcher):");
    if (!repoPath) return;
    localStorage.setItem('gh_repo', repoPath);
  }}

  const tr = document.getElementById('job-row-' + jobId);
  const oldStatus = tr.getAttribute('data-status');
  
  // Optimistic UI update
  tr.setAttribute('data-status', newStatus);
  const tag = tr.querySelector('.status-tag');
  if (tag) {{
    tag.className = 'status-tag status-' + newStatus;
    tag.innerText = newStatus;
  }}
  applyFilters();

  try {{
    // 1. Get current cache file from GitHub API
    const getUrl = `https://api.github.com/repos/${{repoPath}}/contents/jobs_cache.json`;
    const getResp = await fetch(getUrl, {{
      headers: {{ "Authorization": "Bearer " + token, "Accept": "application/vnd.github.v3+json" }}
    }});
    
    if (!getResp.ok) throw new Error("Failed to fetch jobs_cache.json from GitHub API: " + getResp.statusText);
    const fileData = await getResp.json();
    
    // Decode base64 content correctly supporting utf-8
    const rawJson = decodeURIComponent(escape(atob(fileData.content)));
    const cacheObj = JSON.parse(rawJson);

    if (cacheObj[jobId]) {{
      cacheObj[jobId].status = newStatus;
      cacheObj[jobId].updated_at = new Date().toISOString();
    }} else {{
      throw new Error("Job ID not found in cache JSON.");
    }}

    // Encode utf-8 to base64
    const updatedContent = btoa(unescape(encodeURIComponent(JSON.stringify(cacheObj, null, 2))));

    // 2. Commit update back to main
    const putResp = await fetch(getUrl, {{
      method: "PUT",
      headers: {{
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "application/vnd.github.v3+json"
      }},
      body: JSON.stringify({{
        message: `Update job ${{jobId}} status to ${{newStatus}}`,
        content: updatedContent,
        sha: fileData.sha,
        branch: "main"
      }})
    }});

    if (!putResp.ok) throw new Error("Failed to commit change to GitHub.");
    console.log(`Successfully saved status '${{newStatus}}' for job ${{jobId}}`);

  }} catch (err) {{
    alert("Error updating status: " + err.message);
    // Revert UI on error
    tr.setAttribute('data-status', oldStatus);
    applyFilters();
  }}
}}

function sortTable(columnIndex) {{
  var table = document.getElementById("jobTable");
  var rows = Array.from(table.querySelectorAll("tbody tr"));
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

<h1>🎯 Job Matching & Application Tracker</h1>
<p class="sub">Generated on {today} • Phase 2 Interactive Dashboard</p>

<div class="token-bar">
  <span>🔑 <strong>GitHub Token:</strong></span>
  <input type="password" id="ghTokenInput" placeholder="ghp_xxxxxxxxxxxx (Requires repo contents write scope)">
  <button onclick="setGithubToken()">Save Token</button>
</div>

<div class="tabs">
  <button class="tab active" id="tab-new" onclick="filterTab('new')">📥 Inbox (New)</button>
  <button class="tab" id="tab-interested" onclick="filterTab('interested')">⭐ Interested</button>
  <button class="tab" id="tab-applied" onclick="filterTab('applied')">🚀 Applied</button>
  <button class="tab" id="tab-dismissed" onclick="filterTab('dismissed')">❌ Dismissed</button>
  <button class="tab" id="tab-all" onclick="filterTab('all')">📋 All Postings</button>
</div>

<div class="controls">
  <input type="text" id="searchBox" onkeyup="applyFilters()" placeholder="Filter current view by title or company...">
</div>

<table id="jobTable">
<thead>
<tr>
  <th onclick="sortTable(0)" class="desc">Score</th>
  <th onclick="sortTable(1)">Job Title & Details</th>
  <th onclick="sortTable(2)">Company</th>
  <th onclick="sortTable(3)">First Seen / Age</th>
  <th onclick="sortTable(4)">Source</th>
  <th>Triage Action</th>
</tr>
</thead>
<tbody>
"""
    for job in scored_jobs:
        job_id = job["id"]
        status = job.get("status", "new")
        age_display = job.get('posted') if job.get('posted') else compute_posting_age(job.get('first_seen'))
        
        html += f"""
<tr id="job-row-{job_id}" data-status="{status}">
  <td><span class="score-badge">{job.get('score', 0.0):.2f}</span></td>
  <td>
    <div><span class="status-tag status-{status}">{status}</span></div>
    <strong>{job['title']}</strong><br>
    <a href="{job['url']}" target="_blank" class="btn-link" style="margin-top: 4px;">View Posting ↗</a>
  </td>
  <td>{job['company']}</td>
  <td>{age_display}</td>
  <td>{job['source']}</td>
  <td>
    <div class="btn-group">
      <button title="Mark Interested" class="btn-action {'active-interested' if status=='interested' else ''}" onclick="updateJobStatus('{job_id}', 'interested')">⭐</button>
      <button title="Mark Applied" class="btn-action {'active-applied' if status=='applied' else ''}" onclick="updateJobStatus('{job_id}', 'applied')">🚀</button>
      <button title="Dismiss Posting" class="btn-action {'active-dismissed' if status=='dismissed' else ''}" onclick="updateJobStatus('{job_id}', 'dismissed')">❌</button>
      <button title="Reset to Inbox" class="btn-action {'active-new' if status=='new' else ''}" onclick="updateJobStatus('{job_id}', 'new')">📥</button>
    </div>
  </td>
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

    # Update cache with newly discovered items, maintaining user status if already present
    new_jobs_count = 0
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    for j in discovered_jobs:
        job_id = j["id"]
        if job_id not in cache:
            j["first_seen"] = today_str
            j["status"] = "new"  # Phase 2 default status
            cache[job_id] = j
            new_jobs_count += 1
        else:
            # Preserve user status and first_seen date while updating title/url/body_text
            existing_status = cache[job_id].get("status", "new")
            existing_first_seen = cache[job_id].get("first_seen", today_str)
            cache[job_id].update(j)
            cache[job_id]["status"] = existing_status
            cache[job_id]["first_seen"] = existing_first_seen

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
    print(f"[INFO] Complete! Dashboard updated at docs/current.html with {len(all_jobs)} total listings.")

if __name__ == "__main__":
    main()
