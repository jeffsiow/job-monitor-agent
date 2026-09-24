import os
import json
import hashlib
import time
import requests
import re
import yaml
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import CrossEncoder

BASE_DIR = Path(__file__).parent
EXPERIENCE_PATH = BASE_DIR / "experience_library.md"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
CACHE_PATH = BASE_DIR / "jobs_cache.json"
SOURCES_PATH = BASE_DIR / "sources.yaml"
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Preferred locations for scoring (Calgary-centric + remote-ok)
PREFERRED_LOCATIONS = [
    "calgary", "alberta", "ab", "canada", "remote", "hybrid",
    "work from home", "wfh", "anywhere", "distributed"
]
NEGATIVE_LOCATIONS = [
    "united states only", "us only", "usa only", "on-site only",
    "must be located in", "relocation required"
]


def load_experience_library():
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return (
        "Mechanical Engineer, P.Eng., PMP, Project Manager, Program Manager, "
        "Engineering Manager, Operations Manager, CleanTech, Energy, "
        "Industrial Software, Digital Twin, SCADA, IIoT, Process Engineering."
    )


def load_sources():
    if not SOURCES_PATH.exists():
        print("[WARN] sources.yaml not found – no sources will be fetched.")
        return []
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [s for s in data.get("sources", []) if s.get("enabled", True)]


def clean_string(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
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
# Location scoring helper
# ---------------------------------------------------------

def score_location(text, location_hint=""):
    """Return a score in [-1.0, 1.0] based on location signals."""
    blob = clean_string((text or "") + " " + (location_hint or ""))
    if not blob:
        return 0.0

    score = 0.0
    for kw in PREFERRED_LOCATIONS:
        if kw in blob:
            score += 0.35
    for kw in NEGATIVE_LOCATIONS:
        if kw in blob:
            score -= 0.5

    # Strong boosts
    if "calgary" in blob:
        score += 0.4
    if "remote" in blob or "work from home" in blob or "wfh" in blob:
        score += 0.3
    if "canada" in blob or "alberta" in blob:
        score += 0.2

    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------
# Multi-Factor Match Function (now includes location)
# ---------------------------------------------------------

def compute_composite_scores(jobs, cache_data, model):
    base_experience = load_experience_library()[:1800]

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

    target_role_keywords = [
        "project manager", "program manager", "operations manager",
        "project engineer", "engineering manager", "lead", "director",
        "principal", "senior engineer", "solutions architect", "product manager"
    ]

    exp_pairs = [
        [base_experience, job["title"] + " at " + job["company"] + ": " + job.get("body_text", "")]
        for job in jobs
    ]
    raw_exp_scores = model.predict(exp_pairs) if exp_pairs else []

    for idx, job in enumerate(jobs):
        s_exp = float(raw_exp_scores[idx]) if idx < len(raw_exp_scores) else 0.0

        company_clean = clean_string(job.get("company", ""))
        s_company = 0.0
        if company_clean in target_companies:
            s_company = 1.0
        elif company_clean in dismissed_companies:
            s_company = -1.0
        # soft preference for known good companies (even before user marks them)
        elif any(k in company_clean for k in [
            "eavor", "kanin", "seeq", "black & veatch", "city of calgary",
            "aveva", "cognite", "c3.ai", "symphony", "sight machine",
            "flowmingo", "apex", "siemens", "autodesk", "hexagon"
        ]):
            s_company = 0.4

        title_clean = clean_string(job.get("title", ""))
        s_role = 0.0
        if any(kw in title_clean for kw in target_role_keywords):
            s_role += 0.5
        if any(ref in title_clean or title_clean in ref for ref in interested_role_titles):
            s_role += 0.5
        if any(ref in title_clean or title_clean in ref for ref in dismissed_role_titles):
            s_role -= 0.5

        # NEW: location factor
        loc_text = " ".join([
            job.get("location", ""),
            job.get("body_text", ""),
            job.get("title", ""),
            job.get("company", "")
        ])
        s_loc = score_location(loc_text, job.get("location_hint", ""))

        # Weights (sum ≈ 1.0)
        w_exp, w_company, w_role, w_loc = 0.50, 0.15, 0.15, 0.20
        final_score = (
            w_exp * s_exp +
            w_company * s_company +
            w_role * s_role +
            w_loc * s_loc
        )
        job["score"] = round(final_score, 3)
        job["loc_score"] = round(s_loc, 3)  # optional debug


# ---------------------------------------------------------
# Generic helpers used by multiple fetchers
# ---------------------------------------------------------

def _make_job(source_id, company, title, url, body_text="", location="", location_hint=""):
    return {
        "id": make_job_id(source_id, title, url, company),
        "source": source_id,
        "title": title.strip(),
        "company": company,
        "url": url,
        "location": location or "",
        "location_hint": location_hint or "",
        "body_text": (title + " at " + company + ": " + (body_text or location or ""))[:2000],
        "posted": "Recent"
    }


# ---------------------------------------------------------
# Source-specific fetchers
# ---------------------------------------------------------

def fetch_bamboohr(source):
    jobs = []
    url = source["urls"][0]
    company = source["company"]
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
        if resp.status_code == 200:
            for result in resp.json().get("result", []):
                title = result.get("jobOpeningName", "").strip()
                job_id_num = result.get("id", "")
                link = f"https://eavortechnologies.bamboohr.com/careers/{job_id_num}"
                if title:
                    jobs.append(_make_job(
                        source["id"], company, title, link,
                        location_hint=source.get("location_hint", "")
                    ))
            print(f"[INFO] {company}: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} BambooHR failed: {e}")
    return jobs


def fetch_workable(source):
    jobs = []
    url = source["urls"][0]
    company = source["company"]
    try:
        resp = requests.post(
            url,
            json={"query": "", "location": [], "department": []},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = item.get("title", "").strip()
                shortcode = item.get("shortcode", "")
                link = f"https://apply.workable.com/seeq/j/{shortcode}/"
                loc = ", ".join(
                    l.get("city") or l.get("country") or ""
                    for l in item.get("locations", [])
                )
                if title:
                    jobs.append(_make_job(
                        source["id"], company, title, link,
                        location=loc,
                        location_hint=source.get("location_hint", "")
                    ))
            print(f"[INFO] {company}: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} Workable failed: {e}")
    return jobs


def fetch_workday(source):
    """Generic Workday CXS API scraper (POST /jobs + optional detail)."""
    jobs = []
    company = source["company"]
    base_url = source["urls"][0].rstrip("/")
    # Derive the CXS endpoint from the careers URL
    # e.g. https://aveva.wd3.myworkdayjobs.com/en-GB/AVEVA_careers
    #   → https://aveva.wd3.myworkdayjobs.com/wday/cxs/aveva/AVEVA_careers
    try:
        parsed = urlparse(base_url)
        host = parsed.netloc
        path_parts = [p for p in parsed.path.split("/") if p]
        # last non-empty segment is usually the site name
        site = path_parts[-1] if path_parts else "External"
        tenant = host.split(".")[0]
        cxs_base = f"https://{host}/wday/cxs/{tenant}/{site}"
    except Exception:
        print(f"[WARN] Could not parse Workday URL for {company}")
        return jobs

    try:
        offset = 0
        limit = 20
        while True:
            payload = {
                "appliedFacets": source.get("workday_facets", {}),
                "limit": limit,
                "offset": offset,
                "searchText": ""
            }
            resp = requests.post(
                f"{cxs_base}/jobs",
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                timeout=25
            )
            if resp.status_code != 200:
                print(f"[WARN] Workday list failed for {company}: {resp.status_code}")
                break
            data = resp.json()
            postings = data.get("jobPostings", [])
            if not postings:
                break
            for p in postings:
                title = p.get("title", "").strip()
                external_path = p.get("externalPath", "")
                locations = p.get("locationsText", "") or p.get("bulletFields", [""])[0]
                link = urljoin(base_url + "/", external_path.lstrip("/"))
                if title:
                    jobs.append(_make_job(
                        source["id"], company, title, link,
                        location=locations,
                        body_text=locations,
                        location_hint=source.get("location_hint", "")
                    ))
            total = data.get("total", 0)
            offset += limit
            if offset >= total or len(postings) < limit:
                break
            time.sleep(0.4)
        print(f"[INFO] {company} (Workday): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} Workday failed: {e}")
    return jobs


def fetch_kanin(source):
    jobs = []
    url = source["urls"][0]
    company = source["company"]
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.json():
            content = resp.json()[0].get("content", {}).get("rendered", "")
            soup = BeautifulSoup(content, "html.parser")
            for el in soup.find_all(["h2", "h3", "h4", "p", "a"]):
                text = el.get_text(strip=True)
                if any(k in text.lower() for k in ["engineer", "manager", "director", "lead", "developer", "specialist"]):
                    link = el.get("href") if el.name == "a" else "https://kaninenergy.com/careers/"
                    jobs.append(_make_job(
                        source["id"], company, text, link,
                        location_hint=source.get("location_hint", "")
                    ))
            print(f"[INFO] {company}: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} failed: {e}")
    return jobs


def fetch_playwright_generic(source):
    """
    Generic Playwright scraper used for most of the new target companies.
    Tries common job-card / link patterns and extracts title + href + nearby location text.
    """
    jobs = []
    company = source["company"]
    location_hint = source.get("location_hint", "")
    click_view_all = source.get("click_view_all", False)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for url in source["urls"]:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    time.sleep(3)

                    if click_view_all:
                        # C3.ai and similar – try common “View All” / “See all jobs” buttons
                        for selector in [
                            "button:has-text('View All')",
                            "a:has-text('View All')",
                            "button:has-text('See all')",
                            "[data-testid*='view-all']",
                            "text=View all open positions"
                        ]:
                            try:
                                btn = page.locator(selector).first
                                if btn.is_visible(timeout=2000):
                                    btn.click()
                                    time.sleep(2)
                                    break
                            except Exception:
                                continue

                    # Scroll a bit to trigger lazy loading
                    for _ in range(3):
                        page.mouse.wheel(0, 2000)
                        time.sleep(1)

                    content = page.content()
                    soup = BeautifulSoup(content, "html.parser")

                    # Broad set of link patterns that usually indicate a job posting
                    candidates = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        title = a.get_text(strip=True)
                        if not title or len(title) < 4:
                            continue
                        low_href = href.lower()
                        low_title = title.lower()
                        if any(x in low_href for x in ["/job/", "/jobs/", "/careers/", "/position/", "/opening/", "requisition", "apply"]):
                            if "search" not in low_title and "filter" not in low_title:
                                candidates.append((title, href))

                    # Also look for common job-card containers
                    for card in soup.select("[class*='job'], [class*='position'], [class*='opening'], [data-job], li"):
                        a = card.find("a", href=True)
                        if not a:
                            continue
                        title = a.get_text(strip=True) or card.get_text(strip=True)[:120]
                        href = a["href"]
                        if title and len(title) > 4:
                            candidates.append((title, href))

                    seen = set()
                    for title, href in candidates:
                        full_url = href if href.startswith("http") else urljoin(url, href)
                        key = (clean_string(title), normalize_url(full_url))
                        if key in seen:
                            continue
                        seen.add(key)

                        # Try to find a location nearby in the DOM
                        loc = ""
                        # This is best-effort; the location scorer will also look at body_text
                        jobs.append(_make_job(
                            source["id"], company, title, full_url,
                            location=loc,
                            location_hint=location_hint
                        ))

                except Exception as e:
                    print(f"[WARN] Playwright page error for {company} ({url}): {e}")

            browser.close()
            print(f"[INFO] {company} (Playwright): Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} Playwright failed: {e}")
    return jobs


def fetch_city_of_calgary(source):
    """Special-case for the PeopleSoft-style City of Calgary portal."""
    jobs = []
    url = source["urls"][0]
    company = source["company"]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page.goto(url, wait_until="networkidle", timeout=45000)
            time.sleep(5)

            for frame in page.frames:
                try:
                    soup = BeautifulSoup(frame.content(), "html.parser")
                    for a in soup.find_all(["a", "span"]):
                        title = a.get_text(strip=True)
                        if any(k in title.lower() for k in [
                            "engineer", "manager", "planner", "analyst", "lead",
                            "coordinator", "officer", "specialist", "project"
                        ]):
                            jobs.append(_make_job(
                                source["id"], company, title, url,
                                location_hint=source.get("location_hint", "Calgary")
                            ))
                except Exception:
                    continue
            browser.close()
            print(f"[INFO] {company}: Found {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] {company} failed: {e}")
    return jobs

def fetch_airtable_climatetech(source):
    """
    Fetches records from Airtable's public view endpoint and applies custom location/remote filters:
    ("Job Location" Contains "Calgary") OR ("Country" Contains "United States of America" AND "Remote" Does Not Contain "Onsite Only")
    """
    jobs = []
    url = source["urls"][0]
    
    # Extract shared View ID and Table ID from the URL
    m = re.search(r"/(shr[A-Za-z0-9]+)/(tbl[A-Za-z0-9]+)", url)
    if not m:
        print(f"[WARN] Could not parse Airtable view/table IDs from URL: {url}")
        return jobs

    share_id, table_id = m.group(1), m.group(2)
    api_url = f"https://airtable.com/v0.3/table/{table_id}/read"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "x-airtable-inter-service-client": "webClient",
        "x-requested-with": "XMLHttpRequest"
    }

    offset = None
    total_fetched = 0

    try:
        while True:
            params = {
                "requestId": "reqFetcher",
                "shareLinkId": share_id
            }
            if offset:
                params["offset"] = offset

            resp = requests.get(api_url, headers=headers, params=params, timeout=25)
            if resp.status_code != 200:
                print(f"[WARN] Airtable read failed with status: {resp.status_code}")
                break

            data = resp.json().get("data", {})
            rows = data.get("rows", [])
            if not rows:
                break

            for row in rows:
                cell_values = row.get("cellValuesByColumnId", {})
                
                # Stringify all row cell values for filter checks
                row_str_map = {}
                for k, v in cell_values.items():
                    if isinstance(v, list):
                        row_str_map[k] = " ".join([str(x) for x in v])
                    else:
                        row_str_map[k] = str(v) if v is not None else ""

                full_row_text = " ".join(row_str_map.values())

                # Extract title & URL heuristics
                job_url = ""
                job_title = ""
                company_name = source.get("company", "Climate Tech List")

                for k, v in cell_values.items():
                    if isinstance(v, str) and v.startswith("http"):
                        job_url = v
                    elif isinstance(v, dict) and "url" in v:
                        job_url = v["url"]

                title_candidates = [v for v in cell_values.values() if isinstance(v, str) and not v.startswith("http") and len(v) > 2]
                if title_candidates:
                    job_title = title_candidates[0]

                # Applied Filters:
                # ("Job Location" Contains "Calgary") OR ("Country" Contains "United States of America" AND "Remote" Does Not Contain "Onsite Only")
                has_calgary = "calgary" in full_row_text.lower()
                has_usa = "united states" in full_row_text.lower() or "usa" in full_row_text.lower()
                is_onsite_only = "onsite only" in full_row_text.lower() or "on-site only" in full_row_text.lower()

                condition_1 = has_calgary
                condition_2 = has_usa and not is_onsite_only

                if condition_1 or condition_2:
                    if job_title:
                        link = job_url if job_url else url
                        jobs.append(_make_job(
                            source["id"],
                            company_name,
                            job_title,
                            link,
                            body_text=full_row_text,
                            location="Calgary / US Remote",
                            location_hint=source.get("location_hint", "")
                        ))

            total_fetched += len(rows)
            offset = data.get("offset")
            if not offset:
                break
            time.sleep(0.3)

        print(f"[INFO] Climate Tech List (Airtable): Processed {total_fetched} rows, matched {len(jobs)} postings.")
    except Exception as e:
        print(f"[WARN] Climate Tech List Airtable fetch failed: {e}")

    return jobs

# ---------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------

FETCHERS = {
    "airtable": fetch_airtable_climatetech,  # <--- ADD THIS LINE
    "bamboohr": fetch_bamboohr,
    "workable": fetch_workable,
    "workday": fetch_workday,
    "requests_json": fetch_kanin,
    "playwright": fetch_playwright_generic,
    "custom": fetch_playwright_generic,
}


def fetch_source(source):
    src_type = source.get("type", "playwright")
    # Special override for City of Calgary
    if source["id"] == "city_of_calgary":
        return fetch_city_of_calgary(source)
    fetcher = FETCHERS.get(src_type, fetch_playwright_generic)
    return fetcher(source)


# ---------------------------------------------------------
# Dashboard (unchanged except for minor polish)
# ---------------------------------------------------------

DASHBOARD_SCRIPT = """
let currentTab = 'new';
const STORAGE_TOKEN_KEY = 'gh_pat';
const STORAGE_REPO_KEY = 'gh_repo';
const STORAGE_PENDING_KEY = 'gh_pending_changes';

function loadPending() {
  try { return JSON.parse(localStorage.getItem(STORAGE_PENDING_KEY) || '{}'); }
  catch (e) { return {}; }
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
  if (token) localStorage.setItem(STORAGE_TOKEN_KEY, token); else localStorage.removeItem(STORAGE_TOKEN_KEY);
  if (repo) localStorage.setItem(STORAGE_REPO_KEY, repo); else localStorage.removeItem(STORAGE_REPO_KEY);
  alert('Saved locally in this browser.');
}
window.onload = function() {
  const savedToken = localStorage.getItem(STORAGE_TOKEN_KEY);
  const savedRepo = localStorage.getItem(STORAGE_REPO_KEY);
  if (savedToken) document.getElementById('ghTokenInput').value = savedToken;
  if (savedRepo) document.getElementById('ghRepoInput').value = savedRepo;
  const pending = loadPending();
  Object.keys(pending).forEach(function(jobId) { applyStatusToRow(jobId, pending[jobId].status); });
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
  if (tag) { tag.className = 'status-tag status-' + status; tag.innerText = status; }
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
      if (putResp.status === 409 && attempt === 0) continue;
      if (!putResp.ok) throw new Error("GitHub write failed: " + putResp.statusText);
      savePending({});
      syncBtn.innerText = 'Synced ✓';
      setTimeout(updatePendingBadge, 1500);
      return;
    } catch (e) {
      alert("Sync failed: " + e.message + ". Your changes are still saved in this browser — try Sync again.");
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
      va = parseFloat(va) || 0; vb = parseFloat(vb) || 0;
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
  th.asc::after { content: " ▲"; font-size: 10px; }
  th.desc::after { content: " ▼"; font-size: 10px; }
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

    loc_display = job.get("location") or job.get("location_hint") or ""
    return (
        f'<tr id="job-row-{job_id}" data-status="{status}">'
        f'<td><span class="score-badge">{job["score"]:.2f}</span></td>'
        f'<td><div><span class="status-tag status-{status}">{status}</span></div>'
        f'<strong>{job["title"]}</strong><br>'
        f'<a href="{job["url"]}" target="_blank" class="btn-link" style="margin-top:4px;">View Posting ↗</a></td>'
        f'<td>{job["company"]}<br><small style="color:#6c757d">{loc_display}</small></td>'
        f'<td>{age_str}</td>'
        f'<td>{job["source"]}</td>'
        f'<td><div class="btn-group">'
        f'<button title="Mark Interested" data-status-btn="interested" class="btn-action {active_class("interested")}" onclick="queueStatusChange(\'{job_id}\', \'interested\')">⭐</button>'
        f'<button title="Mark Applied" data-status-btn="applied" class="btn-action {active_class("applied")}" onclick="queueStatusChange(\'{job_id}\', \'applied\')">🚀</button>'
        f'<button title="Dismiss Posting" data-status-btn="dismissed" class="btn-action {active_class("dismissed")}" onclick="queueStatusChange(\'{job_id}\', \'dismissed\')">❌</button>'
        f'<button title="Reset to Inbox" data-status-btn="new" class="btn-action {active_class("new")}" onclick="queueStatusChange(\'{job_id}\', \'new\')">📥</button>'
        f'</div></td></tr>'
    )


def render_html_dashboard(ranked_jobs, cache_data, today_str):
    rows_html = "".join(render_job_row(job, cache_data, today_str) for job in ranked_jobs)
    page_title = f"Engineering-Tech Job Matcher — {today_str}"
    if TEMPLATE_PATH.exists():
        page_title = TEMPLATE_PATH.read_text(encoding="utf-8").replace("{{TODAY}}", today_str).strip()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{page_title}</title>
<style>
{DASHBOARD_STYLE}
</style>
<script>
{DASHBOARD_SCRIPT}
</script>
</head>
<body>
<h1>{page_title}</h1>
<div class="sub">{len(ranked_jobs)} postings found (Calgary / Remote Engineering-Tech focus).
Click statuses freely, then hit "Sync Now" — nothing is written to GitHub until you sync.</div>
<div class="token-bar">
<input id="ghTokenInput" type="password" placeholder="GitHub token (repo-scoped, Contents: read/write)">
<input id="ghRepoInput" type="text" placeholder="username/job-matcher">
<button onclick="setGithubConfig()">Save</button>
<button id="syncBtn" onclick="syncNow()">Sync Now (<span id="pendingCount">0</span>)</button>
</div>
<div class="tabs">
<button class="tab" id="tab-new" onclick="filterTab('new')">New</button>
<button class="tab" id="tab-interested" onclick="filterTab('interested')">Interested</button>
<button class="tab" id="tab-applied" onclick="filterTab('applied')">Applied</button>
<button class="tab" id="tab-dismissed" onclick="filterTab('dismissed')">Dismissed</button>
<button class="tab" id="tab-all" onclick="filterTab('all')">All</button>
</div>
<div class="controls">
<input id="searchBox" type="text" placeholder="Search title, company, location..." oninput="applyFilters()">
</div>
<table id="jobTable">
<thead><tr>
<th onclick="sortTable(0, 'number')">Score</th>
<th onclick="sortTable(1, 'text')">Job</th>
<th onclick="sortTable(2, 'text')">Company / Location</th>
<th onclick="sortTable(3, 'text')">Posted</th>
<th onclick="sortTable(4, 'text')">Source</th>
<th>Status</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""
    return html


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    cache_data = load_cache()
    sources = load_sources()

    all_jobs = []
    for source in sources:
        print(f"[INFO] Fetching source: {source['id']} ({source.get('type')})")
        try:
            jobs = fetch_source(source)
            all_jobs.extend(jobs)
        except Exception as e:
            print(f"[WARN] Source {source['id']} raised: {e}")

    # De-duplicate
    seen_ids = set()
    unique_jobs = []
    for job in all_jobs:
        if job["id"] in seen_ids:
            continue
        seen_ids.add(job["id"])
        unique_jobs.append(job)

    print(f"[INFO] Total postings before de-dup: {len(all_jobs)}, after: {len(unique_jobs)}")

    if unique_jobs:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        compute_composite_scores(unique_jobs, cache_data, model)

    ranked_jobs = sorted(unique_jobs, key=lambda j: j.get("score", 0), reverse=True)

    # Update cache
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
    report_path = REPORTS_DIR / f"report-{today_str}.html"
    report_path.write_text(html, encoding="utf-8")
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")

    print(f"[INFO] Dashboard written to {report_path} and docs/index.html")


if __name__ == "__main__":
    main()
