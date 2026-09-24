import os
import json
import hashlib
import time
import requests
import re
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
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"
SOURCES_PATH = BASE_DIR / "sources.yaml"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Role/title keywords used both to (a) decide whether a scraped link looks
# like a real job posting at all, and (b) as part of the role-match score.
ROLE_KEYWORDS = [
    "engineer", "engineering", "manager", "developer", "architect", "scientist",
    "lead", "director", "specialist", "technician", "consultant", "analyst",
    "coordinator", "officer", "programmer", "administrator"
]

# Substrings that mark a posting as Calgary/remote-friendly vs. clearly elsewhere.
LOCATION_OK_KEYWORDS = ["calgary", "alberta", "remote", "distributed", "anywhere", "hybrid", "canada"]


def load_sources():
    if not SOURCES_PATH.exists():
        print("[WARN] sources.yaml not found.")
        return []
    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("sources", [])


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


def score_location(location_text):
    """+1 for Calgary/Alberta/remote-ish text, -1 for some other named location,
    0 (neutral) when we couldn't tell -- most curated sources are already
    filtered to Calgary/remote via their URL, so 'unknown' is not penalized."""
    if not location_text:
        return 0.0
    text = location_text.lower()
    if any(k in text for k in LOCATION_OK_KEYWORDS):
        return 1.0
    return -1.0


# ---------------------------------------------------------
# Multi-Factor Match Function
# ---------------------------------------------------------

def compute_composite_scores(jobs, cache_data, model, known_target_companies):
    base_experience = load_experience_library()[:1500]

    target_companies = set(known_target_companies)
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
        if company_clean in dismissed_companies:
            s_company = -1.0
        elif company_clean in target_companies:
            s_company = 1.0

        title_clean = clean_string(job.get("title", ""))
        s_role = 0.0

        if any(kw in title_clean for kw in target_role_keywords):
            s_role += 0.5

        if any(ref_title in title_clean or title_clean in ref_title for ref_title in interested_role_titles):
            s_role += 0.5

        if any(ref_title in title_clean or title_clean in ref_title for ref_title in dismissed_role_titles):
            s_role -= 0.5

        s_location = score_location(job.get("location_text", ""))

        w_exp, w_company, w_role, w_location = 0.45, 0.15, 0.15, 0.25
        final_score = (w_exp * s_exp) + (w_company * s_company) + (w_role * s_role) + (w_location * s_location)

        job["score"] = round(final_score, 3)


# ---------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------

def fetch_workday_jobs(cfg):
    """Workday's public CXS search API. No browser needed -- this is the same
    endpoint the site's own search box calls. location_ids (if given) are
    passed as an applied facet so the API itself returns pre-filtered results."""
    jobs = []
    company = cfg.get("company", cfg["name"])
    tenant = cfg["tenant"]
    wd_host = cfg["wd_host"]
    site = cfg["site"]
    locale = cfg.get("locale", "en-US")
    location_ids = cfg.get("location_ids", [])

    api_url = "https://" + tenant + "." + wd_host + ".myworkdayjobs.com/wday/cxs/" + tenant + "/" + site + "/jobs"
    job_base_url = "https://" + tenant + "." + wd_host + ".myworkdayjobs.com/" + locale + "/" + site

    payload = {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}
    if location_ids:
        payload["appliedFacets"]["locations"] = location_ids

    try:
        resp = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
        if resp.status_code == 200:
            postings = resp.json().get("jobPostings", [])
            for item in postings:
                title = item.get("title", "").strip()
                path = item.get("externalPath", "")
                link = job_base_url + path
                location_text = item.get("locationsText", "") or item.get("bulletFields", [""])[0] if item.get("bulletFields") else item.get("locationsText", "")
                jobs.append({
                    "id": make_job_id(cfg["name"], title, link, company),
                    "source": cfg["name"],
                    "title": title,
                    "company": company,
                    "url": link,
                    "location_text": location_text or "",
                    "body_text": title + " at " + company + " (" + (location_text or "") + ")"
                })
        else:
            print("[WARN] Workday (" + cfg["name"] + ") returned status " + str(resp.status_code))
    except Exception as e:
        print("[WARN] Workday fetch failed for " + cfg["name"] + ": " + str(e))

    print("[INFO] " + cfg["name"] + " (Workday): Found " + str(len(jobs)) + " postings.")
    return jobs


# Text labels that keep showing up as false positives from nav bars, cookie
# banners, and language switchers on marketing-heavy corporate career pages --
# excluded even if their href happens to match a job-like pattern.
NAV_TEXT_BLACKLIST = {
    "leadership", "about", "about us", "contact", "contact us", "careers", "career",
    "company", "capabilities", "solutions", "solutions overview", "documentation",
    "glossary", "transportation", "manufacturing", "utilities", "federal", "experience",
    "delivery", "our leadership", "find a partner", "partners", "generative ai",
    "ai overview", "search", "apply", "apply now", "apply here", "apply today",
    "home", "sign in", "sign up", "log in", "login",
    "privacy policy", "terms", "cookie policy", "cookie settings", "manage cookies",
    "français", "english", "deutsch", "português", "nederlands", "italiano", "español",
    "our story", "our mission", "press", "news", "blog", "events", "resources",
}

# A job posting URL almost always encodes a specific listing: a numeric/GUID-ish
# job id, or a path segment naming the ATS's job route. Plain nav/marketing
# links (About, Leadership, /careers/manufacturing-overview, language switches,
# hubspot/marketo/linkedin tracking pixels) don't match this.
JOB_URL_PATTERN = re.compile(
    r'(/job/|/jobs/[\w\-]{3,}|/job-|/position/|/posting/|/vacanc\w*/|/opening/'
    r'|req(uisition)?[_\-]?id?=|jobid=|jr[_\-]?\d|gh_jid=|/jobs\?.*\bid=|icims\.com.*/jobs/\d)',
    re.IGNORECASE
)


def _looks_like_job_link(href, text):
    if not text:
        return False
    t = text.strip()
    if len(t) < 6 or len(t) > 140:
        return False
    if t.lower() in NAV_TEXT_BLACKLIST:
        return False
    if len(t.split()) < 2:
        return False
    return bool(JOB_URL_PATTERN.search((href or "").lower()))


def _extract_title_text(anchor_tag):
    """Prefer a heading/title-ish element inside the link over the anchor's
    full text -- on card-style listings the anchor often wraps the title
    PLUS the location, work mode, req id, department and a duplicated
    'Apply Now' label with no separators, which otherwise all get glued
    into one string."""
    heading = anchor_tag.find(["h1", "h2", "h3", "h4", "h5", "strong"])
    if heading:
        txt = heading.get_text(strip=True)
        if txt:
            return txt
    title_el = anchor_tag.find(attrs={"class": re.compile(r"title", re.IGNORECASE)})
    if title_el:
        txt = title_el.get_text(strip=True)
        if txt:
            return txt
    return anchor_tag.get_text(strip=True)


# Card text on some ATS boards (seen on iCIMS-hosted pages) comes through as
# one duplicated blob, e.g.:
#   "Apply Now Forward Deployment Engineer Req ID: 3078 Location US home
#    Remote Department Professional Services Apply Now"
# This pulls the real title and location back out of that pattern.
CARD_TEXT_PATTERN = re.compile(
    r'^(?:Apply Now\s*)?(?P<title>.+?)\s*Req\.?\s*ID:?\s*\S+\s*Location:?\s*'
    r'(?P<location>.+?)\s*(?:Department|Apply Now|$)',
    re.IGNORECASE
)


def _clean_card_text(raw_text, fallback_location=""):
    """Returns (title, location_text). Falls back to the raw text as the
    title (stripped of a leading/trailing 'Apply Now') if the structured
    pattern above doesn't match."""
    m = CARD_TEXT_PATTERN.match(raw_text.strip())
    if m:
        title = m.group("title").strip(" -")
        location = m.group("location").strip(" -")
        if title:
            return title, (location or fallback_location)

    cleaned = re.sub(r'^(Apply Now)\s*', '', raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*(Apply Now)$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip(), fallback_location


# Some ATS "Apply Now" links (seen on iCIMS) end in /login (an application
# flow) rather than the public job posting page. Best-effort swap to the
# page people can actually open without an account.
ICIMS_LOGIN_SUFFIX_RE = re.compile(r'/(login|apply)/?(\?.*)?$', re.IGNORECASE)


def _normalize_job_url(url):
    if "icims.com" in url.lower() and ICIMS_LOGIN_SUFFIX_RE.search(url):
        return ICIMS_LOGIN_SUFFIX_RE.sub('/job', url)
    return url


def _nearby_text(anchor_tag, max_len=300):
    """Best-effort location text: walk up a couple of ancestor containers and
    grab their text, since job boards rarely put location in the <a> itself."""
    node = anchor_tag
    collected = []
    for _ in range(3):
        if node is None or node.parent is None:
            break
        node = node.parent
        txt = node.get_text(" ", strip=True)
        if txt:
            collected.append(txt)
        if len(" ".join(collected)) > max_len:
            break
    return " ".join(collected)[:max_len]



def _extract_jobposting_jsonld(html, base_url):
    """Many ATS/career pages embed schema.org JobPosting structured data
    (JSON-LD) for SEO -- when present, it's far more reliable than scraping
    visible links, since it can't be confused with nav/marketing content."""
    results = []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return results

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        entries = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("@graph"), list):
                entries.extend(item["@graph"])
            else:
                entries.append(item)

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("@type")
            is_job = entry_type == "JobPosting" or (isinstance(entry_type, list) and "JobPosting" in entry_type)
            if not is_job:
                continue

            title = (entry.get("title") or "").strip()
            if not title:
                continue

            url = entry.get("url") or (entry.get("hiringOrganization") or {}).get("sameAs") or base_url
            if not str(url).startswith("http"):
                url = requests.compat.urljoin(base_url, str(url))

            location_text = ""
            loc = entry.get("jobLocation")
            if isinstance(loc, list):
                loc = loc[0] if loc else {}
            if isinstance(loc, dict):
                addr = loc.get("address", {})
                if isinstance(addr, dict):
                    location_text = ", ".join(filter(None, [
                        addr.get("addressLocality"), addr.get("addressRegion"), addr.get("addressCountry")
                    ]))
            if entry.get("jobLocationType") == "TELECOMMUTE":
                location_text = (location_text + " Remote").strip(", ")

            results.append({"title": title, "url": url, "location_text": location_text})

    return results


def fetch_playwright_generic(cfg):
    """Generic scraper for a careers page: loads the url (optionally clicking a
    'view all'-style button first), then tries schema.org JobPosting structured
    data first (most reliable), falling back to anchors that look like job
    postings by URL pattern. Best-effort by nature -- sites that render
    results via unusual JS or hide them behind auth/captchas may return few
    or zero results and will need a follow-up look at the Action logs."""
    jobs = []
    company = cfg.get("company", cfg["name"])
    urls = cfg.get("urls") or [cfg.get("url")]
    click_text = cfg.get("click_button_text")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

            for url in urls:
                if not url:
                    continue
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=40000)
                    time.sleep(4)

                    if click_text:
                        try:
                            page.get_by_text(click_text, exact=False).first.click(timeout=5000)
                            time.sleep(3)
                        except Exception:
                            print("[INFO] " + cfg["name"] + ": click_button_text '" + click_text + "' not found/clickable, continuing without it.")

                    html = page.content()

                    jsonld_jobs = _extract_jobposting_jsonld(html, url)
                    if jsonld_jobs:
                        for jd in jsonld_jobs:
                            jd_url = _normalize_job_url(jd["url"])
                            jobs.append({
                                "id": make_job_id(cfg["name"], jd["title"], jd_url, company),
                                "source": cfg["name"],
                                "title": jd["title"],
                                "company": company,
                                "url": jd_url,
                                "location_text": jd["location_text"],
                                "body_text": jd["title"] + " at " + company + " (" + jd["location_text"] + ")"
                            })
                        print("[INFO] " + cfg["name"] + ": used JobPosting structured data (" + str(len(jsonld_jobs)) + " postings) for " + url)
                        continue

                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        raw_text = a.get_text(strip=True)
                        if not _looks_like_job_link(href, raw_text):
                            continue
                        raw_title = _extract_title_text(a)
                        nearby = _nearby_text(a)
                        title, location_text = _clean_card_text(raw_title, fallback_location=nearby)
                        if not title or title.lower() in NAV_TEXT_BLACKLIST or len(title.split()) < 2:
                            continue
                        full_url = href if href.startswith("http") else requests.compat.urljoin(url, href)
                        full_url = _normalize_job_url(full_url)
                        jobs.append({
                            "id": make_job_id(cfg["name"], title, full_url, company),
                            "source": cfg["name"],
                            "title": title,
                            "company": company,
                            "url": full_url,
                            "location_text": location_text,
                            "body_text": title + " at " + company + " (" + location_text + ")"
                        })
                except Exception as e:
                    print("[WARN] " + cfg["name"] + " failed on " + url + ": " + str(e))

            browser.close()
    except Exception as e:
        print("[WARN] " + cfg["name"] + " Playwright session failed: " + str(e))

    print("[INFO] " + cfg["name"] + " (Playwright, generic): Found " + str(len(jobs)) + " postings.")
    return jobs


def fetch_bamboohr(cfg):
    jobs = []
    company = cfg.get("company", "Eavor Technologies")
    url = "https://eavortechnologies.bamboohr.com/careers/list"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        if resp.status_code == 200:
            for result in resp.json().get("result", []):
                title = result.get("jobOpeningName", "").strip()
                job_id_num = result.get("id", "")
                location_text = result.get("location", {}).get("city", "") if isinstance(result.get("location"), dict) else ""
                link = "https://eavortechnologies.bamboohr.com/careers/" + str(job_id_num)
                jobs.append({
                    "id": make_job_id(cfg["name"], title, link, company),
                    "source": cfg["name"],
                    "title": title,
                    "company": company,
                    "url": link,
                    "location_text": location_text,
                    "body_text": title + " - " + company + " Calgary"
                })
            print("[INFO] " + cfg["name"] + ": Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] " + cfg["name"] + " API failed: " + str(e))
    return jobs


def fetch_workable(cfg):
    jobs = []
    company = cfg.get("company", "Seeq")
    url = "https://apply.workable.com/api/v3/accounts/seeq/jobs"
    try:
        resp = requests.post(url, json={"query": "", "location": [], "department": []}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = item.get("title", "").strip()
                shortcode = item.get("shortcode", "")
                location_text = item.get("location", {}).get("city", "") if isinstance(item.get("location"), dict) else ""
                link = "https://apply.workable.com/seeq/j/" + shortcode + "/"
                jobs.append({
                    "id": make_job_id(cfg["name"], title, link, company),
                    "source": cfg["name"],
                    "title": title,
                    "company": company,
                    "url": link,
                    "location_text": location_text,
                    "body_text": title + " - " + company + " Careers"
                })
            print("[INFO] " + cfg["name"] + ": Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] " + cfg["name"] + " API failed: " + str(e))
    return jobs


def fetch_kanin_energy(cfg):
    jobs = []
    company = cfg.get("company", "Kanin Energy")
    url = "https://kaninenergy.com/wp-json/wp/v2/pages?slug=careers"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.json()) > 0:
            content = resp.json()[0].get("content", {}).get("rendered", "")
            soup = BeautifulSoup(content, "html.parser")
            elements = soup.find_all(["h2", "h3", "h4", "p", "a"])
            for el in elements:
                text = el.get_text(strip=True)
                if any(k in text.lower() for k in ROLE_KEYWORDS):
                    link = el.get("href") if el.name == "a" else "https://kaninenergy.com/careers/"
                    jobs.append({
                        "id": make_job_id(cfg["name"], text, link, company),
                        "source": cfg["name"],
                        "title": text,
                        "company": company,
                        "url": link,
                        "location_text": "Calgary",
                        "body_text": text + " at " + company
                    })
            print("[INFO] " + cfg["name"] + ": Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] " + cfg["name"] + " fetch failed: " + str(e))
    return jobs


def fetch_city_of_calgary(cfg):
    jobs = []
    company = cfg.get("company", "City of Calgary")
    url = "https://recruiting.calgary.ca/psc/hcm/EMPLOYEE/HRMS/c/HRS_HRAM_FL.HRS_CG_SEARCH_FL.GBL?Page=HRS_APP_SCHJOB_FL&Action=U"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page.goto(url, wait_until="networkidle", timeout=40000)
            time.sleep(5)

            for frame in page.frames:
                try:
                    soup = BeautifulSoup(frame.content(), "html.parser")
                    for a in soup.find_all(["a", "span"]):
                        title = a.get_text(strip=True)
                        if any(k in title.lower() for k in ROLE_KEYWORDS):
                            jobs.append({
                                "id": make_job_id(cfg["name"], title, url, company),
                                "source": cfg["name"],
                                "title": title,
                                "company": company,
                                "url": url,
                                "location_text": "Calgary",
                                "body_text": title + " - " + company + " Careers"
                            })
                except Exception:
                    continue

            browser.close()
            print("[INFO] " + cfg["name"] + ": Found " + str(len(jobs)) + " postings.")
    except Exception as e:
        print("[WARN] " + cfg["name"] + " Playwright fetch failed: " + str(e))
    return jobs


TYPE_HANDLERS = {
    "workday": fetch_workday_jobs,
    "playwright": fetch_playwright_generic,
    "bamboohr": fetch_bamboohr,
    "workable": fetch_workable,
    "wordpress_json": fetch_kanin_energy,
    "psft_iframe": fetch_city_of_calgary,
}


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

function clearBrowserData() {
  if (!confirm('This clears your saved GitHub token, repo name, and any un-synced status clicks from this browser. Continue?')) {
    return;
  }
  localStorage.removeItem(STORAGE_TOKEN_KEY);
  localStorage.removeItem(STORAGE_REPO_KEY);
  localStorage.removeItem(STORAGE_PENDING_KEY);
  location.reload();
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
  .clear-link { margin-left: auto; font-size: 12px; color: #856404; cursor: pointer; text-decoration: underline; background: none; border: none; padding: 0; }

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
  .location-tag { font-size: 11px; color: #6c757d; }
"""


def render_job_row(job, cache_data, today_str):
    job_id = job["id"]
    status = cache_data.get(job_id, {}).get("status", "new")
    first_seen = cache_data.get(job_id, {}).get("first_seen", today_str)
    age_str = compute_posting_age(first_seen)
    location_text = job.get("location_text", "") or "—"

    def active_class(btn_status):
        return "active-" + btn_status if status == btn_status else ""

    return (
        '<tr id="job-row-' + job_id + '" data-status="' + status + '">'
        '<td><span class="score-badge">' + format(job["score"], ".2f") + '</span></td>'
        '<td><div><span class="status-tag status-' + status + '">' + status + '</span></div>'
        '<strong>' + job["title"] + '</strong><br>'
        '<span class="location-tag">' + location_text + '</span><br>'
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
        '<button class="clear-link" onclick="clearBrowserData()">Clear saved token / pending changes</button>'
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
    sources = load_sources()

    known_target_companies = set(clean_string(s.get("company", "")) for s in sources if s.get("company"))

    all_jobs = []
    for cfg in sources:
        handler = TYPE_HANDLERS.get(cfg.get("type"))
        if not handler:
            print("[WARN] No handler for source type '" + str(cfg.get("type")) + "' (" + cfg.get("name", "?") + ")")
            continue
        try:
            all_jobs.extend(handler(cfg))
        except Exception as e:
            print("[WARN] Source '" + cfg.get("name", "?") + "' raised an unexpected error: " + str(e))

    # De-duplicate: keep the first occurrence of each unique job id.
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
        compute_composite_scores(unique_jobs, cache_data, model, known_target_companies)

    ranked_jobs = sorted(unique_jobs, key=lambda j: j["score"], reverse=True)

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
