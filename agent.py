import os
import json
import hashlib
import time
import requests
import re
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
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Job titles searched via Google Jobs (SerpAPI). Edit this list to change
# what gets searched for.
GOOGLE_JOB_QUERIES = [
    "Project Manager",
    "Program Manager",
    "Operations Manager",
    "Project Engineer",
    "Engineering Manager",
]


def load_experience_library():
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return "Mechanical Engineer, P.Eng., PMP, Project Manager, CleanTech, Energy."


def clean_string(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def normalize_url(url):
    if not url:
        return ""
    url = url.split("?")[0].split("#")[0]
    return url.rstrip("/").lower()


def make_job_id(source_name, title, url, company=""):
    norm_title = clean_string(title)
    norm_company = clean_string(company)
    norm_url = normalize_url(url)
    raw = clean_string(source_name) + "|" + norm_company + "|" + norm_title + "|" + norm_url
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("[WARN] Failed to read jobs_cache.json: " + str(e))
    return {}


def save_cache(cache_data):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("[WARN] Failed to write jobs_cache.json: " + str(e))


def compute_posting_age(first_seen_str):
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
            return str(delta) + " days ago"
    except Exception:
        return first_seen_str


# ---------------------------------------------------------
# Multi-Factor Match Function
# ---------------------------------------------------------

def compute_composite_scores(jobs, cache_data, model):
    base_experience = load_experience_library()[:1500]

    target_companies = set()
    dismissed_companies = set()
    interested_role_titles = []
    dismissed_role_titles = []

    for job_id, job in cache_data.items():
        status = job.get("status")
        if status in ("interested", "applied"):
            if job.get("company"):
                target_companies.add(clean_string(job["company"]))
            if job.get("title"):
                interested_role_titles.append(clean_string(job["title"]))
        elif status == "dismissed":
            if job.get("company"):
                dismissed_companies.add(clean_string(job["company"]))
            if job.get("title"):
                dismissed_role_titles.append(clean_string(job["title"]))

    target_role_keywords = ["project manager", "project engineer", "operations", "lead", "director"]

    exp_pairs = [[base_experience, job['title'] + " at " + job['company'] + ": " + job['body_text']] for job in jobs]
    raw_exp_scores = model.predict(exp_pairs)

    for idx, job in enumerate(jobs):
        s_exp = float(raw_exp_scores[idx])

        company_clean = clean_string(job.get("company", ""))
        s_company = 0.0
        if company_clean in target_companies:
            s_company = 1.0
        elif company_clean in dismissed_companies:
            s_company = -1.0
        elif any(k in company_clean for k in ["eavor", "kanin", "seeq", "black & veatch", "city of calgary"]):
            s_company = 0.5

        title_clean = clean_string(job.get("title", ""))
        s_role = 0.0

        if any(kw in title_clean for kw in target_role_keywords):
            s_role += 0.5

        if any(ref_title in title_clean or title_clean in ref_title for ref_title in interested_role_titles):
            s_role += 0.5

        if any(ref_title in title_clean or title_clean in ref_title for ref_title in dismissed_role_titles):
            s_role -= 0.5

        w_exp, w_company, w_role = 0.60, 0.20, 0.20
        final_score = (w_exp * s_exp) + (w_company * s_company) + (w_role * s_role)

        job["score"] = round(final_score, 3)


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
            "q": query + " " + location,
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

                    # SerpAPI's job_id is a stable identifier for a given posting.
                    # share_link is NOT stable -- it carries a per-request locale
                    # and tracking token that changes on every call, even for the
                    # exact same job. Hashing on share_link was causing the same
                    # posting to show up as two (or more) separate rows whenever
                    # it matched more than one search query. Use job_id for
                    # identity, keep share_link only as the clickable url.
                    stable_id = item.get("job_id", "")

                    if title and link:
                        jobs.append({
                            "id": make_job_id("google_jobs", title, stable_id or link, company),
                            "source": "google_jobs",
                            "title": title.strip(),
                            "company": company.strip(),
                            "url": link,
                            "body_text": title + " at " + company + ": " + desc[:1500],
                            "posted": item.get("detected_extensions", {}).get("posted_at", "Past week")
                        })
        except Exception as e:
            print("[WARN] Google Jobs batch failed for '" + query + "' (start=" + str(start) + "): " + str(e))

    print("[INFO] Google Jobs ('" + query + "'): Found " + str(len(jobs)) + " postings.")
    return jobs


def fetch_bamboohr_eavor():
    jobs = []
    url = "https://eavortechnologies.bamboohr.com/careers/list"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        if resp.status_code == 200:
            for result in resp.json().get("result", []):
                title = result.get("jobOpeningName", "").strip()
                job_id_num = result.get("id", "")
                link = "https://eavortechnologies.bamboohr.com/careers/" + str(job_id_num)
                jobs.append({
                    "id": make_job_id("eavor", title, link, "Eavor Technologies"),
                    "source": "eavor",
                    "title": title,
                    "company": "Eavor Technologies",
                    "url": link,
                    "body_text": title + " - Eavor Technologies Calgary"
                })
            print("[INFO] Eavor: Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] Eavor API failed: " + str(e))
    return jobs


def fetch_workable_seeq():
    jobs = []
    url = "https://apply.workable.com/api/v3/accounts/seeq/jobs"
    try:
        resp = requests.post(url, json={"query": "", "location": [], "department": []}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = item.get("title", "").strip()
                shortcode = item.get("shortcode", "")
                link = "https://apply.workable.com/seeq/j/" + shortcode + "/"
                jobs.append({
                    "id": make_job_id("seeq", title, link, "Seeq"),
                    "source": "seeq",
                    "title": title,
                    "company": "Seeq",
                    "url": link,
                    "body_text": title + " - Seeq Careers"
                })
            print("[INFO] Seeq: Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] Seeq API failed: " + str(e))
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
                        full_url = href if href.startswith("http") else "https://careers.bv.com" + href
                        jobs.append({
                            "id": make_job_id("black_veatch", title, full_url, "Black & Veatch"),
                            "source": "black_veatch",
                            "title": title,
                            "company": "Black & Veatch",
                            "url": full_url,
                            "body_text": title + " - Black & Veatch Project Management"
                        })
            browser.close()
            print("[INFO] Black & Veatch (Playwright): Found " + str(len(jobs)) + " total postings across both URLs.")
    except Exception as e:
        print("[WARN] Black & Veatch Playwright fetch failed: " + str(e))
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
                        "id": make_job_id("kanin_energy", text, link, "Kanin Energy"),
                        "source": "kanin_energy",
                        "title": text,
                        "company": "Kanin Energy",
                        "url": link,
                        "body_text": text + " at Kanin Energy"
                    })
            print("[INFO] Kanin Energy: Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] Kanin Energy fetch failed: " + str(e))
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
                                "id": make_job_id("city_of_calgary", title, url, "City of Calgary"),
                                "source": "city_of_calgary",
                                "title": title,
                                "company": "City of Calgary",
                                "url": url,
                                "body_text": title + " - City of Calgary Careers"
                            })
                except Exception:
                    continue

            browser.close()
            print("[INFO] City of Calgary (Playwright): Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] City of Calgary Playwright fetch failed: " + str(e))
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
                            full_url = href if href.startswith("http") else "https://www.climatetechlist.com" + href
                            jobs.append({
                                "id": make_job_id("climate_tech_list", title, full_url, "Climate Tech List"),
                                "source": "climate_tech_list",
                                "title": title,
                                "company": "Climate Tech List",
                                "url": full_url,
                                "body_text": title + " - Climate Tech List Calgary"
                            })
                except Exception:
                    continue

            browser.close()
            print("[INFO] Climate Tech List (Playwright): Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] Climate Tech List Playwright fetch failed: " + str(e))
    return jobs


# ---------------------------------------------------------
# Report Writer & HTML Dashboard Generator
# ---------------------------------------------------------

DASHBOARD_SCRIPT = """
let currentTab = 'new';
const STORAGE_TOKEN_KEY = 'gh_pat';
const STORAGE_REPO_KEY = 'gh_repo';
const STORAGE_PENDING_KEY = 'gh_pending_changes';

function loadPending() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_PENDING_KEY) || '{}');
  } catch (e) {
    return {};
  }
}

function savePending(pending) {
  localStorage.setItem(STORAGE_PENDING_KEY, JSON.stringify(pending));
  updatePendingBadge();
}

function updatePendingBadge() {
  const pending = loadPending();
  const count = Object.keys(pending).length;
  const badge = document.getElementById('pendingCount');
  const syncBtn = document.getElementById('syncBtn');
  if (badge) badge.innerText = count;
  if (syncBtn) syncBtn.style.display = count > 0 ? 'inline-block' : 'none';
}

function setGithubConfig() {
  const token = document.getElementById('ghTokenInput').value.trim();
  const repo = document.getElementById('ghRepoInput').value.trim();
  if (token) { localStorage.setItem(STORAGE_TOKEN_KEY, token); } else { localStorage.removeItem(STORAGE_TOKEN_KEY); }
  if (repo) { localStorage.setItem(STORAGE_REPO_KEY, repo); } else { localStorage.removeItem(STORAGE_REPO_KEY); }
  alert('Saved locally in this browser.');
}

window.onload = function() {
  const savedToken = localStorage.getItem(STORAGE_TOKEN_KEY);
  const savedRepo = localStorage.getItem(STORAGE_REPO_KEY);
  if (savedToken) document.getElementById('ghTokenInput').value = savedToken;
  if (savedRepo) document.getElementById('ghRepoInput').value = savedRepo;

  const pending = loadPending();
  Object.keys(pending).forEach(function(jobId) {
    applyStatusToRow(jobId, pending[jobId].status);
  });

  updatePendingBadge();
  filterTab('new');
};

function filterTab(tabName) {
  currentTab = tabName;
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  document.getElementById('tab-' + tabName).classList.add('active');
  applyFilters();
}

function applyFilters() {
  const searchInput = document.getElementById("searchBox").value.toLowerCase();
  const rows = document.querySelectorAll("#jobTable tbody tr");
  rows.forEach(function(row) {
    const rowStatus = row.getAttribute("data-status") || "new";
    const rowText = row.innerText.toLowerCase();
    const matchesTab = (currentTab === 'all') || (rowStatus === currentTab);
    const matchesSearch = rowText.includes(searchInput);
    row.style.display = (matchesTab && matchesSearch) ? "" : "none";
  });
}

function applyStatusToRow(jobId, status) {
  const tr = document.getElementById('job-row-' + jobId);
  if (!tr) return;
  tr.setAttribute('data-status', status);
  const tag = tr.querySelector('.status-tag');
  if (tag) {
    tag.className = 'status-tag status-' + status;
    tag.innerText = status;
  }
  tr.querySelectorAll('.btn-action').forEach(function(btn) {
    btn.classList.remove('active-new', 'active-interested', 'active-applied', 'active-dismissed');
  });
  const activeBtn = tr.querySelector('[data-status-btn="' + status + '"]');
  if (activeBtn) activeBtn.classList.add('active-' + status);
}

function queueStatusChange(jobId, status) {
  const pending = loadPending();
  pending[jobId] = { status: status, updated_at: new Date().toISOString() };
  savePending(pending);
  applyStatusToRow(jobId, status);
  applyFilters();
}

async function syncNow() {
  const token = localStorage.getItem(STORAGE_TOKEN_KEY);
  const repo = localStorage.getItem(STORAGE_REPO_KEY);
  if (!token || !repo) {
    alert("Enter your GitHub token and repo (e.g. username/job-matcher) above and click Save first.");
    return;
  }
  const pending = loadPending();
  const jobIds = Object.keys(pending);
  if (jobIds.length === 0) return;

  const syncBtn = document.getElementById('syncBtn');
  syncBtn.disabled = true;
  syncBtn.innerText = 'Syncing...';

  const apiUrl = 'https://api.github.com/repos/' + repo + '/contents/jobs_cache.json';

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const getResp = await fetch(apiUrl, {
        headers: { "Authorization": "Bearer " + token, "Accept": "application/vnd.github.v3+json" }
      });
      if (!getResp.ok) throw new Error("GitHub read failed: " + getResp.statusText);
      const fileData = await getResp.json();
      const rawJson = decodeURIComponent(escape(atob(fileData.content)));
      const cacheObj = JSON.parse(rawJson);

      jobIds.forEach(function(jobId) {
        if (!cacheObj[jobId]) cacheObj[jobId] = {};
        cacheObj[jobId].status = pending[jobId].status;
        cacheObj[jobId].updated_at = pending[jobId].updated_at;
      });

      const putResp = await fetch(apiUrl, {
        method: "PUT",
        headers: { "Authorization": "Bearer " + token, "Accept": "application/vnd.github.v3+json" },
        body: JSON.stringify({
          message: "Update job status (" + jobIds.length + " change(s)) from dashboard",
          content: btoa(unescape(encodeURIComponent(JSON.stringify(cacheObj, null, 2)))),
          sha: fileData.sha
        })
      });

      if (putResp.status === 409 && attempt === 0) {
        continue;
      }
      if (!putResp.ok) throw new Error("GitHub write failed: " + putResp.statusText);

      savePending({});
      syncBtn.innerText = 'Synced \\u2713';
      setTimeout(updatePendingBadge, 1500);
      return;
    } catch (e) {
      alert("Sync failed: " + e.message + ". Your changes are still saved in this browser -- try Sync again.");
      syncBtn.disabled = false;
      syncBtn.innerText = 'Sync Now';
      return;
    }
  }
  alert("Sync failed after retry (repo changed twice). Try again.");
  syncBtn.disabled = false;
  syncBtn.innerText = 'Sync Now';
}

function sortTable(colIndex, type) {
  const table = document.getElementById('jobTable');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const th = table.querySelectorAll('th')[colIndex];
  const asc = !th.classList.contains('asc');

  table.querySelectorAll('th').forEach(function(h) { h.classList.remove('asc', 'desc'); });
  th.classList.add(asc ? 'asc' : 'desc');

  rows.sort(function(a, b) {
    let va = a.children[colIndex].innerText.trim();
    let vb = b.children[colIndex].innerText.trim();
    if (type === 'number') {
      va = parseFloat(va) || 0;
      vb = parseFloat(vb) || 0;
      return asc ? va - vb : vb - va;
    }
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });

  rows.forEach(function(r) { tbody.appendChild(r); });
}
"""

DASHBOARD_STYLE = """
  body { font-family: system-ui, -apple-system, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }
  h1 { margin-bottom: 5px; color: #1a252f; }
  .sub { color: #6c757d; margin-bottom: 20px; }

  .token-bar { background: #fff3cd; border: 1px solid #ffeeba; padding: 12px 15px; border-radius: 6px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .token-bar input { padding: 6px 10px; border: 1px solid #ced4da; border-radius: 4px; font-size: 13px; }
  .token-bar input#ghTokenInput { flex: 1; min-width: 220px; }
  .token-bar input#ghRepoInput { width: 220px; }
  .token-bar button { padding: 6px 12px; background: #856404; color: white; border: none; border-radius: 4px; cursor: pointer; }
  #syncBtn { background: #0066cc; display: none; }
  #pendingCount { background: #dc3545; color: white; border-radius: 10px; padding: 1px 7px; font-size: 11px; margin-left: 4px; }

  .tabs { display: flex; gap: 8px; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }
  .tab { padding: 8px 16px; border: none; background: #e9ecef; border-radius: 4px; cursor: pointer; font-weight: 600; color: #495057; }
  .tab.active { background: #0066cc; color: white; }

  .controls { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
  #searchBox { padding: 9px 12px; width: 320px; font-size: 14px; border: 1px solid #ced4da; border-radius: 4px; }

  table { border-collapse: collapse; width: 100%; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }
  th { background-color: #0066cc; color: white; cursor: pointer; user-select: none; }
  th:hover { background-color: #0052a5; }
  th.asc::after { content: " \\25B2"; font-size: 10px; }
  th.desc::after { content: " \\25BC"; font-size: 10px; }
  tr:hover { background-color: #f1f5f9; }

  .score-badge { font-weight: bold; padding: 4px 8px; border-radius: 4px; background: #e3f2fd; color: #0d47a1; }

  .btn-group { display: flex; gap: 4px; }
  .btn-action { border: 1px solid #ced4da; background: white; padding: 5px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
  .btn-action:hover { background: #e2e8f0; }
  .btn-action.active-new { background: #e2e8f0; font-weight: bold; }
  .btn-action.active-interested { background: #fff3cd; border-color: #ffeeba; color: #856404; font-weight: bold; }
  .btn-action.active-applied { background: #d4edda; border-color: #c3e6cb; color: #155724; font-weight: bold; }
  .btn-action.active-dismissed { background: #f8d7da; border-color: #f5c6cb; color: #721c24; font-weight: bold; }

  a.btn-link { text-decoration: none; background: #0066cc; color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; display: inline-block; }
  a.btn-link:hover { background: #0052a5; }

  .status-tag { font-size: 11px; padding: 2px 6px; border-radius: 3px; text-transform: uppercase; font-weight: bold; display: inline-block; margin-bottom: 4px; }
  .status-new { background: #e2e8f0; color: #475569; }
  .status-interested { background: #fef08a; color: #854d0e; }
  .status-applied { background: #bbf7d0; color: #166534; }
  .status-dismissed { background: #fecdd3; color: #9f1239; }
"""


def render_job_row(job, cache_data, today_str):
    job_id = job["id"]
    status = cache_data.get(job_id, {}).get("status", "new")
    first_seen = cache_data.get(job_id, {}).get("first_seen", today_str)
    age_str = compute_posting_age(first_seen)

    def active_class(btn_status):
        return "active-" + btn_status if status == btn_status else ""

    return (
        '<tr id="job-row-' + job_id + '" data-status="' + status + '">'
        '<td><span class="score-badge">' + format(job["score"], ".2f") + '</span></td>'
        '<td><div><span class="status-tag status-' + status + '">' + status + '</span></div>'
        '<strong>' + job["title"] + '</strong><br>'
        '<a href="' + job["url"] + '" target="_blank" class="btn-link" style="margin-top: 4px;">View Posting &#8599;</a></td>'
        '<td>' + job["company"] + '</td>'
        '<td>' + age_str + '</td>'
        '<td>' + job["source"] + '</td>'
        '<td><div class="btn-group">'
        '<button title="Mark Interested" data-status-btn="interested" class="btn-action ' + active_class("interested") + '" onclick="queueStatusChange(\'' + job_id + '\', \'interested\')">&#11088;</button>'
        '<button title="Mark Applied" data-status-btn="applied" class="btn-action ' + active_class("applied") + '" onclick="queueStatusChange(\'' + job_id + '\', \'applied\')">&#128640;</button>'
        '<button title="Dismiss Posting" data-status-btn="dismissed" class="btn-action ' + active_class("dismissed") + '" onclick="queueStatusChange(\'' + job_id + '\', \'dismissed\')">&#10060;</button>'
        '<button title="Reset to Inbox" data-status-btn="new" class="btn-action ' + active_class("new") + '" onclick="queueStatusChange(\'' + job_id + '\', \'new\')">&#128229;</button>'
        '</div></td></tr>'
    )


def render_html_dashboard(ranked_jobs, cache_data, today_str):
    rows_html = "".join(render_job_row(job, cache_data, today_str) for job in ranked_jobs)

    if TEMPLATE_PATH.exists():
        title_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        title_template = "Job Matching Dashboard — {{TODAY}}"
    page_title = title_template.replace("{{TODAY}}", today_str).strip()

    html_parts = []
    html_parts.append("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n")
    html_parts.append("<title>" + page_title + "</title>\n")
    html_parts.append("<style>\n" + DASHBOARD_STYLE + "\n</style>\n")
    html_parts.append("<script>\n" + DASHBOARD_SCRIPT + "\n</script>\n")
    html_parts.append("</head>\n<body>\n")
    html_parts.append("<h1>" + page_title + "</h1>\n")
    html_parts.append(
        '<div class="sub">' + str(len(ranked_jobs)) +
        ' postings found. Click statuses freely, then hit &quot;Sync Now&quot; to save '
        '&mdash; nothing is written to GitHub until you sync.</div>\n'
    )
    html_parts.append(
        '<div class="token-bar">'
        '<input id="ghTokenInput" type="password" placeholder="GitHub token (repo-scoped, Contents: read/write)">'
        '<input id="ghRepoInput" type="text" placeholder="username/job-matcher">'
        '<button onclick="setGithubConfig()">Save</button>'
        '<button id="syncBtn" onclick="syncNow()">Sync Now (<span id="pendingCount">0</span>)</button>'
        '</div>\n'
    )
    html_parts.append(
        '<div class="tabs">'
        '<button class="tab" id="tab-new" onclick="filterTab(\'new\')">New</button>'
        '<button class="tab" id="tab-interested" onclick="filterTab(\'interested\')">Interested</button>'
        '<button class="tab" id="tab-applied" onclick="filterTab(\'applied\')">Applied</button>'
        '<button class="tab" id="tab-dismissed" onclick="filterTab(\'dismissed\')">Dismissed</button>'
        '<button class="tab" id="tab-all" onclick="filterTab(\'all\')">All</button>'
        '</div>\n'
    )
    html_parts.append(
        '<div class="controls">'
        '<input id="searchBox" type="text" placeholder="Search title, company..." oninput="applyFilters()">'
        '</div>\n'
    )
    html_parts.append('<table id="jobTable"><thead><tr>')
    html_parts.append('<th onclick="sortTable(0, \'number\')">Score</th>')
    html_parts.append('<th onclick="sortTable(1, \'text\')">Job</th>')
    html_parts.append('<th onclick="sortTable(2, \'text\')">Company</th>')
    html_parts.append('<th onclick="sortTable(3, \'text\')">Posted</th>')
    html_parts.append('<th onclick="sortTable(4, \'text\')">Source</th>')
    html_parts.append('<th>Status</th>')
    html_parts.append('</tr></thead><tbody>')
    html_parts.append(rows_html)
    html_parts.append('</tbody></table>\n</body>\n</html>\n')

    return "".join(html_parts)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    cache_data = load_cache()

    all_jobs = []
    for query in GOOGLE_JOB_QUERIES:
        all_jobs.extend(fetch_google_jobs(query))

    all_jobs.extend(fetch_bamboohr_eavor())
    all_jobs.extend(fetch_workable_seeq())
    all_jobs.extend(fetch_black_veatch_playwright())
    all_jobs.extend(fetch_kanin_energy())
    all_jobs.extend(fetch_city_of_calgary_playwright())
    all_jobs.extend(fetch_climate_tech_list_playwright())

    # De-duplicate: keep the first occurrence of each unique job id. This is a
    # second line of defense on top of the stable-id fix in fetch_google_jobs
    # (e.g. in case the same posting is scraped through two different sources).
    seen_ids = set()
    unique_jobs = []
    for job in all_jobs:
        if job["id"] in seen_ids:
            continue
        seen_ids.add(job["id"])
        unique_jobs.append(job)

    print("[INFO] Total postings before de-dup: " + str(len(all_jobs)) + ", after de-dup: " + str(len(unique_jobs)))

    if unique_jobs:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        compute_composite_scores(unique_jobs, cache_data, model)

    ranked_jobs = sorted(unique_jobs, key=lambda j: j["score"], reverse=True)

    # Update cache: preserve existing status/first_seen for known jobs, add new ones as "new".
    for job in ranked_jobs:
        jid = job["id"]
        if jid not in cache_data:
            cache_data[jid] = {
                "title": job["title"],
                "company": job["company"],
                "status": "new",
                "first_seen": today_str
            }
        else:
            cache_data[jid]["title"] = job["title"]
            cache_data[jid]["company"] = job["company"]

    save_cache(cache_data)

    html = render_html_dashboard(ranked_jobs, cache_data, today_str)

    report_path = REPORTS_DIR / ("report-" + today_str + ".html")
    report_path.write_text(html, encoding="utf-8")

    latest_path = DOCS_DIR / "index.html"
    latest_path.write_text(html, encoding="utf-8")

    print("[INFO] Dashboard written to " + str(report_path) + " and " + str(latest_path))


if __name__ == "__main__":
    main()
