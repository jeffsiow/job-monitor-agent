#!/usr/bin/env python3
"""Daily job matcher for energy / climate-tech project roles.

Fixes vs the broken draft:
  - HTML dashboard uses a template file (no unterminated f-strings)
  - Job identity is company + title, so Google Jobs + career-site copies merge
  - One Playwright browser for every rendered board
  - CrossEncoder logits are sigmoid-normalized before mixing with 0–1 boosts
  - Status / first_seen survive days a posting is missing
  - Heuristic ranker if sentence-transformers is unavailable

Usage:
  python agent.py           # live scrape
  python agent.py --demo    # sample data, no network
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXPERIENCE_PATH = BASE_DIR / "experience_library.md"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
CACHE_PATH = BASE_DIR / "jobs_cache.json"
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"

REPORTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LodeJobMatcher/1.0)"}
TARGET_ROLE_KEYWORDS = [
    "project manager",
    "project engineer",
    "program manager",
    "operations",
    "lead",
    "director",
]
WATCH_COMPANIES = ["eavor", "kanin", "seeq", "black & veatch", "black and veatch", "city of calgary"]
JUNK_TITLE = re.compile(
    r"sign in|log in|privacy|cookie|search jobs|view all|read more|apply now|^home$|^careers$",
    re.I,
)


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_experience_library() -> str:
    if EXPERIENCE_PATH.exists():
        return EXPERIENCE_PATH.read_text(encoding="utf-8")
    return (
        "Mechanical Engineer, P.Eng., PMP, Project Manager, CleanTech, Energy. "
        "Geothermal, waste heat, industrial decarbonization. Calgary."
    )


def clean_string(text) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def normalize_url(url) -> str:
    if not url:
        return ""
    return url.split("?")[0].split("#")[0].rstrip("/").lower()


def identity_key(company: str, title: str) -> str:
    return clean_string(company) + "|" + clean_string(title)


def make_job_id(title: str, url: str = "", company: str = "") -> str:
    key = identity_key(company, title)
    if not clean_string(company):
        key += "|" + normalize_url(url)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print("[WARN] Failed to read jobs_cache.json: " + str(exc))
    return {}


def save_cache(cache_data: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print("[WARN] Failed to write jobs_cache.json: " + str(exc))


def compute_posting_age(first_seen_str: str) -> str:
    if not first_seen_str:
        return "New"
    try:
        first_seen_dt = datetime.strptime(first_seen_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - first_seen_dt).days
        if delta <= 0:
            return "Today"
        if delta == 1:
            return "1 day ago"
        return str(delta) + " days ago"
    except Exception:
        return first_seen_str


def new_job(source, title, company, url, body="", location="", posted="") -> dict:
    title = (title or "").strip()
    company = (company or "").strip()
    return {
        "id": make_job_id(title, url, company),
        "source": source,
        "sources": [source],
        "title": title,
        "company": company,
        "url": url,
        "body_text": body or (title + " at " + company),
        "location": location,
        "posted": posted,
    }


def is_plausible_title(title: str) -> bool:
    t = (title or "").strip()
    if len(t) < 8 or len(t) > 120:
        return False
    if JUNK_TITLE.search(t):
        return False
    return True


def rank_url(url: str) -> int:
    u = (url or "").lower()
    if "google.com" in u or "serpapi" in u:
        return 0
    return 1


def merge_job(a: dict, b: dict) -> dict:
    sources = []
    for src in (a.get("sources") or [a.get("source")]) + (b.get("sources") or [b.get("source")]):
        if src and src not in sources:
            sources.append(src)
    prefer = b if rank_url(b.get("url", "")) > rank_url(a.get("url", "")) else a
    richer = b if len(b.get("body_text") or "") > len(a.get("body_text") or "") else a
    out = dict(a)
    out.update({
        "title": a["title"] if len(a.get("title", "")) >= len(b.get("title", "")) else b["title"],
        "company": a["company"] or b["company"],
        "url": prefer.get("url", a.get("url")),
        "source": sources[0] if sources else a.get("source"),
        "sources": sources,
        "body_text": richer.get("body_text", a.get("body_text")),
        "location": a.get("location") or b.get("location", ""),
        "posted": a.get("posted") or b.get("posted", ""),
    })
    return out


def dedupe_jobs(jobs: list) -> list:
    by_id = {}
    for job in jobs:
        if not job or not job.get("title"):
            continue
        if not is_plausible_title(job["title"]):
            continue
        job["id"] = make_job_id(job["title"], job.get("url", ""), job.get("company", ""))
        job["sources"] = job.get("sources") or ([job["source"]] if job.get("source") else [])
        existing = by_id.get(job["id"])
        by_id[job["id"]] = merge_job(existing, job) if existing else job
    return list(by_id.values())


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def tokenize(text: str) -> set:
    stop = {"the", "and", "for", "with", "from", "that", "this", "are", "have"}
    return {w for w in re.split(r"[^a-z0-9+]+", (text or "").lower()) if len(w) > 2 and w not in stop}


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def heuristic_experience_score(library: str, job: dict) -> float:
    blob = job.get("title", "") + " " + job.get("company", "") + " " + job.get("body_text", "")
    return min(1.0, overlap(tokenize(library[:2500]), tokenize(blob)) * 2.2)


def compute_composite_scores(jobs, cache_data, model=None) -> None:
    if not jobs:
        return
    base_experience = load_experience_library()[:1500]
    target_companies = set()
    interested_role_titles = []
    ignored_companies = set()

    for job in cache_data.values():
        status = job.get("status")
        company = clean_string(job.get("company", ""))
        if status in ("interested", "applied", "to_apply"):
            if company:
                target_companies.add(company)
            if job.get("title"):
                interested_role_titles.append(clean_string(job["title"]))
        if status == "ignored" and company:
            ignored_companies.add(company)

    raw_exp = None
    if model is not None:
        try:
            pairs = [
                [base_experience, job["title"] + " at " + job["company"] + ": " + (job.get("body_text") or "")[:1200]]
                for job in jobs
            ]
            raw_exp = [sigmoid(float(x)) for x in model.predict(pairs)]
        except Exception as exc:
            print("[WARN] Model predict failed, using heuristic: " + str(exc))
            raw_exp = None

    for idx, job in enumerate(jobs):
        s_exp = raw_exp[idx] if raw_exp is not None else heuristic_experience_score(base_experience, job)
        company_clean = clean_string(job.get("company", ""))
        s_company = 0.0
        if company_clean in target_companies:
            s_company = 1.0
        elif any(k in company_clean for k in WATCH_COMPANIES):
            s_company = 0.5
        if company_clean in ignored_companies and company_clean not in target_companies:
            s_company = max(0.0, s_company - 0.35)

        title_clean = clean_string(job.get("title", ""))
        s_role = 0.0
        if any(kw in title_clean for kw in TARGET_ROLE_KEYWORDS):
            s_role += 0.5
        if any(ref and (ref in title_clean or title_clean in ref) for ref in interested_role_titles):
            s_role += 0.5
        s_role = min(1.0, s_role)

        job["score"] = round(0.60 * s_exp + 0.20 * s_company + 0.20 * s_role, 3)
        job["score_parts"] = {
            "experience": round(s_exp, 3),
            "company": round(s_company, 3),
            "role": round(s_role, 3),
        }


def load_model():
    try:
        from sentence_transformers import CrossEncoder

        print("[INFO] Loading CrossEncoder (ms-marco-MiniLM-L-6-v2)…")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as exc:
        print("[WARN] sentence-transformers unavailable (" + str(exc) + "). Using heuristic ranker.")
        return None


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_google_jobs(query, location="Calgary, AB") -> list:
    jobs = []
    if requests is None:
        return jobs
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("[WARN] SERPAPI_KEY missing. Skipping Google Jobs.")
        return jobs
    url = "https://serpapi.com/search.json"
    for start in (0, 10):
        params = {
            "engine": "google_jobs",
            "q": query + " " + location,
            "chips": "date_posted:week",
            "start": start,
            "api_key": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                print("[WARN] Google Jobs HTTP " + str(resp.status_code) + " for " + query)
                continue
            for item in resp.json().get("jobs_results", []):
                title = item.get("title", "")
                company = item.get("company_name", "")
                link = item.get("share_link") or (item.get("related_links") or [{}])[0].get("link", "")
                desc = item.get("description", "")
                if title and link:
                    jobs.append(
                        new_job(
                            "google_jobs",
                            title,
                            company,
                            link,
                            title + " at " + company + ": " + desc[:1500],
                            location,
                            item.get("detected_extensions", {}).get("posted_at", "Past week"),
                        )
                    )
        except Exception as exc:
            print("[WARN] Google Jobs failed for '" + query + "': " + str(exc))
    print("[INFO] Google Jobs ('" + query + "'): " + str(len(jobs)))
    return jobs


def fetch_bamboohr_eavor() -> list:
    jobs = []
    if requests is None:
        return jobs
    url = "https://eavortechnologies.bamboohr.com/careers/list"
    try:
        resp = requests.get(url, headers={"Accept": "application/json", **HEADERS}, timeout=20)
        if resp.status_code == 200:
            for result in resp.json().get("result", []):
                title = (result.get("jobOpeningName") or "").strip()
                job_id_num = result.get("id", "")
                link = "https://eavortechnologies.bamboohr.com/careers/" + str(job_id_num)
                loc = ""
                loc_obj = result.get("location") or {}
                if isinstance(loc_obj, dict):
                    loc = loc_obj.get("city") or loc_obj.get("name") or ""
                jobs.append(new_job("eavor", title, "Eavor Technologies", link, title + " - Eavor Technologies Calgary", loc))
            print("[INFO] Eavor: " + str(len(jobs)))
    except Exception as exc:
        print("[WARN] Eavor API failed: " + str(exc))
    return jobs


def fetch_workable_seeq() -> list:
    jobs = []
    if requests is None:
        return jobs
    url = "https://apply.workable.com/api/v3/accounts/seeq/jobs"
    try:
        resp = requests.post(
            url,
            json={"query": "", "location": [], "department": []},
            headers={**HEADERS, "Accept": "application/json"},
            timeout=20,
        )
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = (item.get("title") or "").strip()
                shortcode = item.get("shortcode", "")
                link = "https://apply.workable.com/seeq/j/" + shortcode + "/"
                loc = ""
                loc_obj = item.get("location") or {}
                if isinstance(loc_obj, dict):
                    loc = loc_obj.get("city") or loc_obj.get("country") or ""
                jobs.append(new_job("seeq", title, "Seeq", link, title + " - Seeq Careers", loc))
            print("[INFO] Seeq: " + str(len(jobs)))
    except Exception as exc:
        print("[WARN] Seeq API failed: " + str(exc))
    return jobs


def fetch_kanin_energy() -> list:
    jobs = []
    if requests is None or BeautifulSoup is None:
        return jobs
    url = "https://kaninenergy.com/wp-json/wp/v2/pages?slug=careers"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200 and resp.json():
            content = resp.json()[0].get("content", {}).get("rendered", "")
            soup = BeautifulSoup(content, "html.parser")
            seen = set()
            for el in soup.find_all(["h2", "h3", "h4", "a"]):
                text = el.get_text(strip=True)
                if not is_plausible_title(text):
                    continue
                if not any(k in text.lower() for k in ("engineer", "manager", "director", "lead", "specialist")):
                    continue
                key = clean_string(text)
                if key in seen:
                    continue
                seen.add(key)
                link = el.get("href") if el.name == "a" and el.get("href") else "https://kaninenergy.com/careers/"
                jobs.append(new_job("kanin_energy", text, "Kanin Energy", link, text + " at Kanin Energy", "Calgary, AB"))
            print("[INFO] Kanin Energy: " + str(len(jobs)))
    except Exception as exc:
        print("[WARN] Kanin Energy failed: " + str(exc))
    return jobs


def scrape_job_links(soup, source, company, base, href_needles) -> list:
    jobs = []
    if soup is None:
        return jobs
    seen = set()
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        href = a["href"]
        if not is_plausible_title(title):
            continue
        if not any(n in href.lower() for n in href_needles):
            continue
        full = href if href.startswith("http") else base.rstrip("/") + "/" + href.lstrip("/")
        key = normalize_url(full) or clean_string(title)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(new_job(source, title, company, full, title + " - " + company))
    return jobs


def run_playwright_boards() -> list:
    jobs = []
    if sync_playwright is None or BeautifulSoup is None:
        print("[WARN] Playwright or BeautifulSoup missing. Skipping rendered boards.")
        return jobs

    boards = [
        {
            "source": "black_veatch",
            "company": "Black & Veatch",
            "base": "https://careers.bv.com",
            "needles": ["/job/"],
            "urls": [
                "https://careers.bv.com/search/?createNewAlert=false&q=project+manager&locationsearch=canada+OR+united+states&optionsFacetsDD_customfield3=&optionsFacetsDD_customfield5=Project+Management",
                "https://careers.bv.com/search/?createNewAlert=false&q=project+manager&locationsearch=canada&optionsFacetsDD_customfield3=&optionsFacetsDD_customfield5=",
            ],
        },
        {
            "source": "cdr_jobs",
            "company": "",
            "base": "https://www.cdrjobs.earth",
            "needles": ["/job/", "/jobs/"],
            "urls": ["https://www.cdrjobs.earth/job-board"],
        },
        {
            "source": "climate_tech_list",
            "company": "",
            "base": "https://www.climatetechlist.com",
            "needles": ["/job/", "/posting", "airtable.com"],
            "urls": ["https://www.climatetechlist.com/jobs?location=calgary"],
        },
    ]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            for board in boards:
                for url in board["urls"]:
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=40000)
                        page.wait_for_timeout(4000)
                        sources = [page] + list(page.frames)
                        found = 0
                        for src in sources:
                            try:
                                soup = BeautifulSoup(src.content(), "html.parser")
                            except Exception:
                                continue
                            batch = scrape_job_links(soup, board["source"], board["company"] or "Unknown", board["base"], board["needles"])
                            for job in batch:
                                if board["source"] == "climate_tech_list" and " at " in job["title"]:
                                    title, company = job["title"].split(" at ", 1)
                                    job["title"] = title.strip()
                                    job["company"] = company.strip()
                                    job["id"] = make_job_id(job["title"], job["url"], job["company"])
                                found += 1
                            jobs.extend(batch)
                        print("[INFO] " + board["source"] + " " + url[:48] + "…: " + str(found) + " raw")
                    except Exception as exc:
                        print("[WARN] " + board["source"] + " fetch failed: " + str(exc))
            try:
                jobs.extend(fetch_city_of_calgary(page))
            except Exception as exc:
                print("[WARN] City of Calgary failed: " + str(exc))
            browser.close()
    except Exception as exc:
        print("[WARN] Playwright session failed: " + str(exc))
    return jobs


def fetch_city_of_calgary(page) -> list:
    jobs = []
    url = "https://recruiting.calgary.ca/psc/hcm/EMPLOYEE/HRMS/c/HRS_HRAM_FL.HRS_CG_SEARCH_FL.GBL?Page=HRS_APP_SCHJOB_FL&Action=U"
    page.goto(url, wait_until="domcontentloaded", timeout=40000)
    page.wait_for_timeout(5000)
    keywords = ("engineer", "manager", "planner", "analyst", "lead", "coordinator", "officer", "specialist", "project")
    seen = set()
    for frame in page.frames:
        try:
            soup = BeautifulSoup(frame.content(), "html.parser")
        except Exception:
            continue
        for a in soup.find_all("a"):
            title = a.get_text(strip=True)
            if not is_plausible_title(title):
                continue
            if not any(k in title.lower() for k in keywords):
                continue
            key = clean_string(title)
            if key in seen:
                continue
            seen.add(key)
            href = a.get("href") or url
            full = href if href.startswith("http") else url
            jobs.append(new_job("city_of_calgary", title, "City of Calgary", full, title + " - City of Calgary Careers", "Calgary, AB"))
    print("[INFO] City of Calgary: " + str(len(jobs)))
    return jobs


# ---------------------------------------------------------------------------
# Demo + reports
# ---------------------------------------------------------------------------

def demo_jobs() -> list:
    samples = [
        ("eavor", "Project Manager, Geothermal Delivery", "Eavor Technologies", "https://eavortechnologies.bamboohr.com/careers/42"),
        ("google_jobs", "Project Manager, Geothermal Delivery", "Eavor Technologies", "https://www.google.com/search?q=eavor+pm"),
        ("kanin_energy", "Project Engineer — Waste Heat to Power", "Kanin Energy", "https://kaninenergy.com/careers/"),
        ("seeq", "Customer Operations Lead", "Seeq", "https://apply.workable.com/seeq/j/OPSLEAD/"),
        ("black_veatch", "Project Manager — Energy", "Black & Veatch", "https://careers.bv.com/job/pm-energy"),
        ("black_veatch", "Project Manager — Energy", "Black & Veatch", "https://careers.bv.com/search/?q=pm"),
        ("city_of_calgary", "Project Manager, Infrastructure", "City of Calgary", "https://recruiting.calgary.ca/"),
        ("climate_tech_list", "Climate Project Manager", "Entropy Inc.", "https://www.climatetechlist.com/jobs/entropy-pm"),
        ("eavor", "Director of Project Delivery", "Eavor Technologies", "https://eavortechnologies.bamboohr.com/careers/18"),
        ("seeq", "Software Engineer, Analytics Platform", "Seeq", "https://apply.workable.com/seeq/j/SWE123/"),
        ("cdr_jobs", "Program Manager, Industrial Decarbonization", "Svante", "https://www.cdrjobs.earth/job/svante-pm"),
    ]
    jobs = []
    for source, title, company, url in samples:
        jobs.append(new_job(source, title, company, url, title + " at " + company + " — Calgary / energy delivery."))
    return jobs


def upsert_cache(jobs: list, cache: dict, today: str) -> None:
    seen_ids = set()
    for job in jobs:
        seen_ids.add(job["id"])
        prev = cache.get(job["id"], {})
        cache[job["id"]] = {
            **prev,
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "url": job["url"],
            "source": job.get("source"),
            "sources": job.get("sources") or [job.get("source")],
            "body_text": job.get("body_text", ""),
            "location": job.get("location", prev.get("location", "")),
            "posted": job.get("posted", prev.get("posted", "")),
            "first_seen": prev.get("first_seen", today),
            "last_seen": today,
            "status": prev.get("status", "new"),
            "score": job.get("score", prev.get("score", 0)),
            "score_parts": job.get("score_parts", prev.get("score_parts", {})),
            "stale": False,
        }
        job["first_seen"] = cache[job["id"]]["first_seen"]
        job["status"] = cache[job["id"]]["status"]

    for job_id, rec in cache.items():
        if job_id not in seen_ids:
            rec["stale"] = True
            if rec.get("status") in ("interested", "to_apply", "applied"):
                jobs.append({
                    "id": job_id,
                    "title": rec.get("title", ""),
                    "company": rec.get("company", ""),
                    "url": rec.get("url", ""),
                    "source": rec.get("source", ""),
                    "sources": rec.get("sources") or [rec.get("source", "")],
                    "body_text": rec.get("body_text", ""),
                    "location": rec.get("location", ""),
                    "posted": rec.get("posted", ""),
                    "first_seen": rec.get("first_seen", today),
                    "status": rec.get("status", "new"),
                    "stale": True,
                })


def render_html_dashboard(ranked_jobs, cache_data, today_str) -> None:
    if TEMPLATE_PATH.exists():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = DEFAULT_TEMPLATE

    rows = []
    for job in ranked_jobs:
        job_id = job["id"]
        rec = cache_data.get(job_id, {})
        status = rec.get("status", job.get("status", "new"))
        first_seen = rec.get("first_seen", job.get("first_seen", today_str))
        age = compute_posting_age(first_seen)
        sources = rec.get("sources") or job.get("sources") or [job.get("source", "")]
        merged = " merged" if len(sources) > 1 else ""
        score = job.get("score", rec.get("score", 0)) or 0
        pct = int(round(float(score) * 100))
        opts = []
        for value, label in (("new", "New"), ("interested", "Interested"), ("to_apply", "To apply"), ("applied", "Applied"), ("ignored", "Ignored")):
            sel = " selected" if status == value else ""
            opts.append('<option value="' + value + '"' + sel + ">" + label + "</option>")
        stale = ' <span class="chip">stale</span>' if rec.get("stale") or job.get("stale") else ""
        rows.append(
            '<tr data-id="' + html.escape(job_id) + '">'
            '<td class="score"><span class="n">' + str(pct) + '</span></td>'
            '<td><a href="' + html.escape(job.get("url") or "#") + '" target="_blank" rel="noopener">'
            + html.escape(job.get("title") or "")
            + "</a>" + stale + '<div class="meta">'
            + html.escape(job.get("company") or "")
            + (" · " + html.escape(job["location"]) if job.get("location") else "")
            + '</div></td>'
            '<td><span class="chip' + merged + '">' + html.escape(", ".join(s for s in sources if s)) + "</span></td>"
            "<td>" + html.escape(age) + "</td>"
            '<td><select onchange="setStatus(\'' + html.escape(job_id) + '\', this.value)">'
            + "".join(opts)
            + "</select></td></tr>"
        )

    active = [j for j in ranked_jobs if cache_data.get(j["id"], {}).get("status") != "ignored"]
    stats = (
        str(len(active)) + " active · "
        + str(sum(1 for j in ranked_jobs if len(j.get("sources") or []) > 1)) + " merged duplicates · "
        + today_str
    )
    html_out = (
        template.replace("{{TODAY}}", html.escape(today_str))
        .replace("{{STATS}}", html.escape(stats))
        .replace("{{ROWS}}", "\n".join(rows) if rows else '<tr><td colspan="5">No postings this run.</td></tr>')
    )
    (DOCS_DIR / "index.html").write_text(html_out, encoding="utf-8")


def write_markdown(ranked_jobs, cache_data, today_str) -> None:
    lines = ["# Job matches — " + today_str, ""]
    for job in ranked_jobs[:40]:
        rec = cache_data.get(job["id"], {})
        if rec.get("status") == "ignored":
            continue
        score = job.get("score", 0) or 0
        sources = ", ".join(job.get("sources") or [job.get("source", "")])
        lines.append(
            "- **{:.0f}** [{}]({}) — {} ({}) {}".format(
                score * 100,
                job.get("title", ""),
                job.get("url") or "#",
                job.get("company", ""),
                sources,
                rec.get("status", "new"),
            )
        )
    (REPORTS_DIR / (today_str + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Lode — {{TODAY}}</title></head>
<body><h1>Job Matching Dashboard — {{TODAY}}</h1><p>{{STATS}}</p>
<table><thead><tr><th>Fit</th><th>Role</th><th>Source</th><th>Age</th><th>Status</th></tr></thead>
<tbody>{{ROWS}}</tbody></table>
<script>
const KEY='lode-status-overrides';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}
function setStatus(id,status){const s=load();s[id]=status;localStorage.setItem(KEY,JSON.stringify(s))}
</script></body></html>
"""


def collect_live() -> list:
    raw = []
    raw += fetch_google_jobs("project manager energy")
    raw += fetch_google_jobs("project engineer geothermal OR cleantech")
    raw += fetch_bamboohr_eavor()
    raw += fetch_workable_seeq()
    raw += fetch_kanin_energy()
    raw += run_playwright_boards()
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use sample jobs, skip the network")
    args = parser.parse_args(argv)

    today = utc_today()
    cache = load_cache()
    raw = demo_jobs() if args.demo else collect_live()
    jobs = dedupe_jobs(raw)
    print("[INFO] Unique after merge: " + str(len(jobs)) + " (from " + str(len(raw)) + " raw)")

    upsert_cache(jobs, cache, today)
    model = None if args.demo else load_model()
    compute_composite_scores(jobs, cache, model)
    for job in jobs:
        if job["id"] in cache:
            cache[job["id"]]["score"] = job.get("score", 0)
            cache[job["id"]]["score_parts"] = job.get("score_parts", {})

    ranked = sorted(jobs, key=lambda j: j.get("score") or 0, reverse=True)
    render_html_dashboard(ranked, cache, today)
    write_markdown(ranked, cache, today)
    save_cache(cache)
    print("[INFO] Wrote docs/index.html, reports/, jobs_cache.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
