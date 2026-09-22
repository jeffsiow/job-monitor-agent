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
    interested_role_titles = []
    
    for job_id, job in cache_data.items():
        if job.get("status") in ["interested", "applied"]:
            if job.get("company"):
                target_companies.add(clean_string(job["company"]))
            if job.get("title"):
                interested_role_titles.append(clean_string(job["title"]))

    target_role_keywords = ["project manager", "project engineer", "operations", "lead", "director"]

    exp_pairs = [[base_experience, job['title'] + " at " + job['company'] + ": " + job['body_text']] for job in jobs]
    raw_exp_scores = model.predict(exp_pairs)

    for idx, job in enumerate(jobs):
        s_exp = float(raw_exp_scores[idx])

        company_clean = clean_string(job.get("company", ""))
        s_company = 0.0
        if company_clean in target_companies:
            s_company = 1.0
        elif any(k in company_clean for k in ["eavor", "kanin", "seeq", "black & veatch", "city of calgary"]):
            s_company = 0.5

        title_clean = clean_string(job.get("title", ""))
        s_role = 0.0
        
        if any(kw in title_clean for kw in target_role_keywords):
            s_role += 0.5
            
        if any(ref_title in title_clean or title_clean in ref_title for ref_title in interested_role_titles):
            s_role += 0.5

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
                    
                    if title and link:
                        jobs.append({
                            "id": make_job_id("google_jobs", title, link, company),
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

def render_html_dashboard(ranked_jobs, cache_data, today_str):
    rows_html = []
    for job in ranked_jobs:
        job_id = job["id"]
        status = cache_data.get(job_id, {}).get("status", "new")
        first_seen = cache_data.get(job_id, {}).get("first_seen", today_str)
        age_str = compute_posting_age(first_seen)

        # Build status dropdown options
        opt_new = 'selected' if status == 'new' else ''
        opt_int = 'selected' if status == 'interested' else ''
        opt_app = 'selected' if status == 'applied' else ''
        opt_ign = 'selected' if status == 'ignored' else ''

        row = (
            f'
