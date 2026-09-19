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
SOURCES_PATH = BASE_DIR / "sources.yaml"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

def load_experience_library():
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return "Mechanical Engineer, P.Eng., PMP, Project Engineering Manager, Energy & CleanTech."

def load_sources():
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("sources", [])
    return []

def make_job_id(source_name, title, url):
    raw = f"{source_name}|{title}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------
# Direct API & Dedicated Parsers (100% Reliable)
# ---------------------------------------------------------

def fetch_google_jobs(query="project engineer", location="Calgary, AB"):
    """Fetches clean job listings via SerpAPI (bypasses Cloudflare entirely)."""
    jobs = []
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("[WARN] SERPAPI_KEY not found in environment. Skipping Google Jobs.")
        return jobs

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "api_key": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
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
                        "posted": item.get("detected_extensions", {}).get("posted_at", "Recent")
                    })
    except Exception as e:
        print(f"[WARN] Google Jobs fetch failed: {e}")
    return jobs

def fetch_bamboohr_eavor():
    jobs = []
    url = "https://eavortechnologies.bamboohr.com/careers/list"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
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
                    "body_text": f"{title} - Calgary, AB",
                    "posted": "Recent"
                })
    except Exception as e:
        print(f"[WARN] Eavor API failed: {e}")
    return jobs

def fetch_workable_seeq():
    jobs = []
    url = "https://apply.workable.com/api/v3/accounts/seeq/jobs"
    try:
        resp = requests.post(url, json={"search": "", "location": []}, timeout=10)
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
    except Exception as e:
        print(f"[WARN] Seeq API failed: {e}")
    return jobs

def fetch_black_veatch():
    jobs = []
    url = "https://careers.bv.com/api/apply/v2/jobs?domain=bv.com&start=0&num=25&q=project%20manager"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code == 200:
            for pos in resp.json().get("positions", []):
                title = pos.get("displayTitle") or pos.get("title", "")
                job_id_num = pos.get("id", "")
                link = f"https://careers.bv.com/job/{job_id_num}"
                if title:
                    jobs.append({
                        "id": make_job_id("black_veatch", title, link),
                        "source": "black_veatch",
                        "title": title,
                        "company": "Black & Veatch",
                        "url": link,
                        "body_text": f"{title} - Engineering & Project Management",
                        "posted": "Recent"
                    })
    except Exception as e:
        print(f"[WARN] Black & Veatch API failed: {e}")
    return jobs

def fetch_kanin_energy():
    """Direct fetch for Kanin Energy WordPress career postings."""
    jobs = []
    url = "https://kaninenergy.com/wp-json/wp/v2/pages?slug=careers"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and len(resp.json()) > 0:
            content = resp.json()[0].get("content", {}).get("rendered", "")
            soup = BeautifulSoup(content, "html.parser")
            # Extract headers or job links inside accordion blocks
            headers = soup.find_all(["h2", "h3", "h4", "a"])
            for h in headers:
                text = h.get_text(strip=True)
                if any(k in text.lower() for k in ["engineer", "manager", "director", "lead", "developer", "analyst"]):
                    link = h.get("href") if h.name == "a" else "https://kaninenergy.com/careers/"
                    jobs.append({
                        "id": make_job_id("kanin_energy", text, link),
                        "source": "kanin_energy",
                        "title": text,
                        "company": "Kanin Energy",
                        "url": link,
                        "body_text": f"{text} at Kanin Energy",
                        "posted": "Recent"
                    })
    except Exception as e:
        print(f"[WARN] Kanin Energy fetch failed: {e}")
    return jobs

def fetch_city_of_calgary():
    """Queries City of Calgary's Oracle Taleo portal."""
    jobs = []
    url = "https://calgary.taleo.net/careersection/2/jobsearch.ftl?lang=en"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            if "jobdetail.ftl?job=" in a["href"].lower() and len(title) > 3:
                full_url = f"https://calgary.taleo.net/careersection/2/{a['href']}"
                jobs.append({
                    "id": make_job_id("city_of_calgary", title, full_url),
                    "source": "city_of_calgary",
                    "title": title,
                    "company": "City of Calgary",
                    "url": full_url,
                    "body_text": f"{title} - City of Calgary Taleo Posting",
                    "posted": "Recent"
                })
    except Exception as e:
        print(f"[WARN] City of Calgary Taleo fetch failed: {e}")
    return jobs

def fetch_cdr_jobs_playwright():
    """Renders Wix Studio list cards on CDR Jobs."""
    jobs = []
    url = "https://www.cdrjobs.earth/job-board"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            time.sleep(3) # Wait for Wix client-side list hydration
            
            soup = BeautifulSoup(page.content(), "html.parser")
            cards = soup.find_all(["div", "li", "a"], class_=lambda c: c and any(x in c.lower() for x in ["job", "list", "item", "card"]))
            
            for card in cards:
                a_tag = card if card.name == "a" else card.find("a")
                if not a_tag:
                    continue
                title = a_tag.get_text(separator=" ", strip=True)
                href = a_tag.get("href", "")
                if title and len(title) > 5 and any(k in title.lower() for k in ["engineer", "manager", "director", "carbon", "lead"]):
                    full_url = f"https://www.cdrjobs.earth{href}" if href.startswith("/") else href
                    jobs.append({
                        "id": make_job_id("cdr_jobs", title, full_url),
                        "source": "cdr_jobs",
                        "title": title,
                        "company": "CDR Earth Board",
                        "url": full_url,
                        "body_text": f"{title} - CDR Earth Posting",
                        "posted": "Recent"
                    })
            browser.close()
    except Exception as e:
        print(f"[WARN] CDR Jobs Playwright fetch failed: {e}")
    return jobs

# ---------------------------------------------------------
# Report Generator
# ---------------------------------------------------------

def write_html_report(scored_jobs):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Job Matching Report — {today}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1a252f; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
  th {{ background-color: #0066cc; color: white; }}
  .score-badge {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; background: #e3f2fd; color: #0d47a1; }}
  a.btn {{ text-decoration: none; background: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
<h1>🎯 Daily Job Match Recommendations ({today})</h1>
<table>
<thead>
<tr><th>Score</th><th>Job Title</th><th>Company</th><th>Source</th><th>Action</th></tr>
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
  <td><a href="{job['url']}" target="_blank" class="btn">View Posting</a></td>
</tr>
"""
    html += "</tbody></table></body></html>"

    with open(REPORTS_DIR / f"report-{today}.html", "w", encoding="utf-8") as f:
        f.write(html)
    with open(DOCS_DIR / "current.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------

def main():
    print("[INFO] Initializing Cross-Encoder Model...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    experience_text = load_experience_library()[:1500]
    
    discovered_jobs = []

    # 1. Fetch Google Jobs via SerpAPI (Replaces Indeed/LinkedIn Scraping)
    print("[INFO] Fetching Google Jobs (Calgary Engineering)...")
    discovered_jobs.extend(fetch_google_jobs("project engineer", "Calgary, AB"))
    discovered_jobs.extend(fetch_google_jobs("project director", "Calgary, AB"))

    # 2. Direct Corporate APIs
    print("[INFO] Fetching Eavor (BambooHR)...")
    discovered_jobs.extend(fetch_bamboohr_eavor())

    print("[INFO] Fetching Seeq (Workable)...")
    discovered_jobs.extend(fetch_workable_seeq())

    print("[INFO] Fetching Black & Veatch...")
    discovered_jobs.extend(fetch_black_veatch())

    print("[INFO] Fetching Kanin Energy...")
    discovered_jobs.extend(fetch_kanin_energy())

    print("[INFO] Fetching City of Calgary (Taleo)...")
    discovered_jobs.extend(fetch_city_of_calgary())

    # 3. Dedicated Playwright Renderers
    print("[INFO] Fetching CDR Jobs...")
    discovered_jobs.extend(fetch_cdr_jobs_playwright())

    print(f"[INFO] Total candidate jobs collected: {len(discovered_jobs)}")

    if discovered_jobs:
        pairs = [[experience_text, f"{j['title']} at {j['company']}: {j['body_text']}"] for j in discovered_jobs]
        scores = model.predict(pairs)

        for job, score in zip(discovered_jobs, scores):
            job["score"] = float(score)

        discovered_jobs.sort(key=lambda x: x["score"], reverse=True)

    write_html_report(discovered_jobs)
    print(f"[INFO] Complete! Output saved to docs/current.html with {len(discovered_jobs)} active listings.")

if __name__ == "__main__":
    main()
