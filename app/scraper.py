"""Fetch Flipkart reviews from a product link.

Handles full links, app-share/short links (dl.flipkart.com/s/..., fkrt.it/...), links without
'pid', and extracts reviews using several strategies (embedded JSON, CSS classes, text markers).
If a plain HTTP fetch is blocked or returns a JavaScript-only page, it falls back to a real
browser when Playwright is installed:  pip install playwright && playwright install chromium

NOTE: Flipkart has no public reviews API and may block bots or change its layout. If nothing is
found, the fetched page is saved to debug_last_response.html in the project folder.
Check Flipkart's Terms of Use before scraping at scale.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from bs4 import BeautifulSoup


class ScrapeError(Exception):
    pass


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.flipkart.com/",
    "Upgrade-Insecure-Requests": "1",
}
# CSS selectors for the review body (Flipkart's class names change; add new ones here)
BODY_SELECTORS = ["div.ZmyHeo", "div.t-ZTKy", "div._6K-7Co", "div.qwjRop", "div.cPHDOP div.row > div > div"]
SHORT_HOSTS = {"fkrt.it", "fkrt.cc", "fkrt.to"}
DEBUG_FILE = Path(__file__).resolve().parent.parent / "debug_last_response.html"


def _is_flipkart(host: str) -> bool:
    return host == "flipkart.com" or host.endswith(".flipkart.com")


# ----------------------------------------------------------------------------- URL handling
def _resolve(client, url: str):
    """Return (reviews_base_url, params, product_name) for any Flipkart link."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if not (_is_flipkart(host) or host in SHORT_HOSTS):
        raise ScrapeError("Please paste a link from flipkart.com.")

    html = ""
    # Short/share links, or links missing the product id: open them to find the real product URL
    if not re.search(r"/itm[0-9A-Za-z]+", p.path) or "pid" not in parse_qs(p.query):
        try:
            r = client.get(url)
        except httpx.HTTPError as e:
            raise ScrapeError(f"Couldn't open that link: {e}")
        p = urlparse(str(r.url))
        html = r.text or ""
        if not _is_flipkart((p.hostname or "").lower()):
            raise ScrapeError("That link doesn't lead to a Flipkart page.")

    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    itm = re.search(r"/(itm[0-9A-Za-z]+)", p.path) or re.search(r"/(itm[0-9A-Za-z]{8,})", html)
    pid = q.get("pid")
    if not pid and html:
        m = re.search(r"[?&;]pid=([A-Z0-9]{10,25})", html) or re.search(r'"productId"\s*:\s*"([A-Z0-9]{10,25})"', html)
        pid = m.group(1) if m else None
    if not itm or not pid:
        raise ScrapeError("Couldn't find the product ID in that link. Open the product page on flipkart.com "
                          "and copy the full link from the address bar.")

    segs = [s for s in p.path.split("/") if s]
    slug = next((s for s in segs if s not in ("dl", "s", "p", "product-reviews") and not s.startswith("itm")), "product")
    params = {"pid": pid, "marketplace": q.get("marketplace", "FLIPKART")}
    if q.get("lid"):
        params["lid"] = q["lid"]
    return f"https://www.flipkart.com/{slug}/product-reviews/{itm.group(1)}", params, slug.replace("-", " ").title()


# ----------------------------------------------------------------------------- parsing
_NOISE = re.compile(
    r"certified buyer|verified purchase|read more|^helpful|permalink|report abuse|^\d+(\.\d+)?$|"
    r"^\d+\s+\w+\s+ago$|^\w{3,9},?\s+\d{4}$|^(yes|no)$", re.I)


def _body_text(el) -> str:
    t = re.sub(r"\s*READ MORE\s*$", "", el.get_text(" ", strip=True), flags=re.I).strip()
    return t if len(t) > 2 else ""


def _from_json(soup):
    """Reviews embedded in window.__INITIAL_STATE__ or JSON-LD (works even when HTML is JS-rendered)."""
    blobs = []
    for s in soup.find_all("script"):
        txt = s.string or s.get_text() or ""
        if s.get("type") == "application/ld+json":
            blobs.append(txt)
        elif "__INITIAL_STATE__" in txt and "{" in txt:
            blobs.append(txt[txt.index("{", txt.index("__INITIAL_STATE__")):])
    out = []
    for b in blobs:
        try:
            data, _ = json.JSONDecoder().raw_decode(b.strip())
        except ValueError:
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                t = x.get("reviewBody") or (x.get("text") if "rating" in x else None)
                if isinstance(t, str) and len(t.strip()) > 2:
                    out.append(re.sub(r"\s*READ MORE\s*$", "", t.strip(), flags=re.I))
                else:
                    stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return out[::-1] if out else out  # stack pops last-first; restore page order


def _from_certified_marker(soup):
    """Last resort: each review block has a 'Certified Buyer' line; take the longest text in its block."""
    out = []
    for m in soup.find_all(string=re.compile(r"Certified Buyer", re.I)):
        node, best = m.parent, None
        while node is not None and node.parent is not None:
            if len(node.find_all(string=re.compile(r"Certified Buyer", re.I))) > 1:
                break
            best, node = node, node.parent
        if best is None:
            continue
        cands = [t.strip() for t in best.stripped_strings if len(t.strip()) > 2 and not _NOISE.search(t.strip())]
        if cands:
            out.append(max(cands, key=len))
    return out


def parse_reviews(html: str):
    soup = BeautifulSoup(html, "lxml")
    out = _from_json(soup)
    if out:
        return out
    for sel in BODY_SELECTORS:
        out = [t for t in (_body_text(e) for e in soup.select(sel)) if t]
        if out:
            return out
    out = []
    for s in soup.find_all(string=re.compile(r"^\s*READ MORE\s*$")):
        if s.parent is not None and s.parent.parent is not None:
            t = _body_text(s.parent.parent)
            if t:
                out.append(t)
    return out or _from_certified_marker(soup)


# ----------------------------------------------------------------------------- fetching
def _http_pages(client, base, params, max_pages, delay, diag):
    for page in range(1, max_pages + 1):
        try:
            r = client.get(base, params={**params, "page": page})
        except httpx.HTTPError as e:
            diag["error"] = str(e)
            return
        diag["status"], diag["html"] = r.status_code, r.text
        if r.status_code != 200:
            return
        if "captcha" in r.text[:5000].lower():
            diag["blocked"] = True
            return
        yield r.text
        time.sleep(delay)


def _browser_pages(base, params, max_pages):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        pg = browser.new_context(user_agent=HEADERS["User-Agent"], locale="en-IN").new_page()
        try:
            for page in range(1, max_pages + 1):
                pg.goto(f"{base}?{urlencode({**params, 'page': page})}", wait_until="domcontentloaded", timeout=30000)
                pg.wait_for_timeout(2500)
                yield pg.content()
        finally:
            browser.close()


def _collect(pages):
    reviews, seen = [], set()
    for html in pages:
        batch = parse_reviews(html)
        key = tuple(batch[:3])
        if not batch or key in seen:  # empty or repeated page => no more reviews
            break
        seen.add(key)
        reviews.extend(batch)
    return reviews


def fetch_reviews(url: str, max_pages: int = 15, delay: float = 0.8):
    diag = {}
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=25) as client:
        try:
            client.get("https://www.flipkart.com/")  # warm-up: picks up cookies like a normal visit
        except httpx.HTTPError:
            pass
        base, params, name = _resolve(client, url)
        reviews = _collect(_http_pages(client, base, params, max_pages, delay, diag))

    if not reviews:  # optional real-browser fallback
        try:
            reviews = _collect(_browser_pages(base, params, max_pages))
        except Exception:
            reviews = []

    if not reviews:
        try:
            DEBUG_FILE.write_text(diag.get("html", ""), encoding="utf-8")
        except OSError:
            pass
        why = ("Flipkart's bot protection blocked the request" if diag.get("blocked")
               else f"Flipkart returned HTTP {diag['status']}" if diag.get("status", 200) != 200
               else f"network error: {diag['error']}" if diag.get("error")
               else "the page had no readable reviews (it may be JavaScript-only or its layout changed)")
        raise ScrapeError(f"Couldn't get reviews: {why}. Install Playwright for browser mode "
                          "(pip install playwright && playwright install chromium), or use 'paste reviews'. "
                          "The fetched page was saved to debug_last_response.html.")
    return name, reviews
