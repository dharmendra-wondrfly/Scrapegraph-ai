"""
Discover candidate program-related URLs: sitemap, homepage links, optional shallow BFS.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from scrapegraphai.docloaders.chromium import ChromiumLoader

_NS_STRIP = re.compile(r"\{[^}]+\}")

NON_HTML_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
)


def _is_html_url(url: str) -> bool:
    """Reject obvious non-HTML asset URLs (PDFs, Office, archives, images)."""
    path = urlparse(url).path.lower()
    return not path.endswith(NON_HTML_EXTENSIONS)


PROGRAM_PATH_KEYWORDS: tuple[str, ...] = (
    "program",
    "class",
    "classes",
    "lesson",
    "lessons",
    "camp",
    "youth",
    "junior",
    "kids",
    "child",
    "tennis",
    "academy",
    "instruction",
    "registration",
    "schedule",
)

# Paths that almost never carry program data and that dilute the LLM's signal.
# A URL whose path contains any of these AND zero positive keywords is dropped.
NEGATIVE_PATH_KEYWORDS: tuple[str, ...] = (
    "blog",
    "news",
    "press",
    "careers",
    "career",
    "jobs",
    "job-",
    "privacy",
    "terms",
    "legal",
    "cookie",
    "login",
    "signin",
    "sign-in",
    "account",
    "cart",
    "checkout",
    "wp-content",
    "wp-admin",
    "/tag/",
    "/category/",
    "/author/",
    "/feed",
    "/rss",
    "sitemap",
)

# Paths we want to surface to a *provider-profile* prompt, separate from program URLs.
AUX_PATH_KEYWORDS: tuple[str, ...] = (
    "about",
    "contact",
    "location",
    "locations",
    "team",
    "staff",
    "who-we-are",
    "our-story",
)


def _strip_ns(tag: str) -> str:
    return _NS_STRIP.sub("", tag)


def _canonical_base(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    p = urlparse(url)
    scheme = p.scheme or "https"
    netloc = p.netloc or ""
    if not netloc:
        raise ValueError(f"Invalid URL (missing host): {url}")
    return f"{scheme}://{netloc}"


def _host_key(netloc: str) -> str:
    return netloc.lower().removeprefix("www.")


def _same_site(url: str, base_netloc: str) -> bool:
    host = urlparse(url).netloc
    if not host:
        return False
    return _host_key(host) == _host_key(base_netloc)


def _score_url(url: str) -> int:
    path = urlparse(url).path.lower()
    return sum(1 for kw in PROGRAM_PATH_KEYWORDS if kw in path)


def _score_components(url: str) -> tuple[int, int]:
    """Return (positive_kw_hits, negative_kw_hits) for a URL path."""
    path = urlparse(url).path.lower()
    pos = sum(1 for kw in PROGRAM_PATH_KEYWORDS if kw in path)
    neg = sum(1 for kw in NEGATIVE_PATH_KEYWORDS if kw in path)
    return pos, neg


def _is_aux_url(url: str) -> bool:
    """True when the URL looks like an about/contact/location page."""
    path = urlparse(url).path.lower()
    return any(kw in path for kw in AUX_PATH_KEYWORDS)


def _fetch_html_playwright(url: str, *, headless: bool, loader_kwargs: dict) -> str | None:
    try:
        loader = ChromiumLoader([url], headless=headless, **loader_kwargs)
        docs = loader.load()
        if docs and getattr(docs[0], "page_content", None):
            return docs[0].page_content
    except Exception:
        return None
    return None


def _extract_hrefs(html: str, base_url: str, base_netloc: str, only_internal: bool) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    invalid_schemes = (
        "mailto:",
        "tel:",
        "javascript:",
        "data:",
        "#",
    )
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(invalid_schemes):
            continue
        full = href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
        frag = full.split("#", 1)[0]
        if not frag.startswith(("http://", "https://")):
            continue
        if only_internal and not _same_site(frag, base_netloc):
            continue
        if not _is_html_url(frag):
            continue
        out.append(frag.rstrip("/"))
    return out


def _fetch_sitemap_urls(base: str, timeout: float = 20.0) -> list[str]:
    parsed = urlparse(base)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    candidates = [
        f"{scheme}://{netloc}/sitemap.xml",
        f"{scheme}://{netloc}/sitemap_index.xml",
    ]
    found: list[str] = []
    for sm_url in candidates:
        try:
            req = Request(sm_url, headers={"User-Agent": "Mozilla/5.0 ScrapeGraphAI-youth-discovery"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except Exception:
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        tag_local = _strip_ns(root.tag).lower()
        if tag_local == "sitemapindex":
            for loc in root.iter():
                if _strip_ns(loc.tag).lower() == "loc" and loc.text:
                    found.extend(_fetch_plain_sitemap(loc.text.strip(), timeout))
        else:
            found.extend(_parse_urlset(root))
        if found:
            break
    return found


def _fetch_plain_sitemap(sm_url: str, timeout: float) -> list[str]:
    try:
        req = Request(sm_url, headers={"User-Agent": "Mozilla/5.0 ScrapeGraphAI-youth-discovery"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:
        return []
    return _parse_urlset(root)


def _parse_urlset(root: ET.Element) -> list[str]:
    urls: list[str] = []
    for el in root.iter():
        if _strip_ns(el.tag).lower() == "loc" and el.text:
            urls.append(el.text.strip())
    return urls


def discover_program_urls(
    base_url: str,
    *,
    max_urls: int = 40,
    seed_depth: int = 2,
    headless: bool = True,
    loader_kwargs: dict | None = None,
    include_base_in_output: bool = True,
) -> list[str]:
    """
    Collect same-site URLs from sitemap.xml (when available), homepage anchors,
    and optional shallow internal crawl (BFS up to ``seed_depth``).

    URLs are ranked by heuristic keyword hits in the path, deduped, capped at ``max_urls``.
    """
    loader_kwargs = loader_kwargs or {}
    base = _canonical_base(base_url).rstrip("/")
    parsed = urlparse(base)
    base_netloc = parsed.netloc

    collected: set[str] = set()

    # Sitemap (filtered to site)
    for u in _fetch_sitemap_urls(base):
        clean_sm = u.split("#", 1)[0].rstrip("/")
        if _same_site(u, base_netloc) and _is_html_url(clean_sm):
            collected.add(clean_sm)

    # Homepage + shallow internal BFS
    if seed_depth > 0:

        def _norm(u: str) -> str:
            return u.split("#", 1)[0].rstrip("/")

        frontier: deque[tuple[str, int]] = deque([(_norm(base), 0)])
        queued: set[str] = {_norm(base)}

        while frontier:
            current, depth = frontier.popleft()
            fetch_url = current + "/" if not current.endswith("/") else current
            page_html = _fetch_html_playwright(
                fetch_url, headless=headless, loader_kwargs=loader_kwargs
            )
            if not page_html:
                continue
            links = _extract_hrefs(page_html, fetch_url, base_netloc, only_internal=True)
            for link in links:
                clean = _norm(link)
                if not _is_html_url(clean):
                    continue
                collected.add(clean)
                if depth + 1 < seed_depth and clean not in queued:
                    queued.add(clean)
                    frontier.append((clean, depth + 1))

    home = base.rstrip("/")
    collected.add(home)

    def _is_pure_noise(u: str) -> bool:
        """Drop URLs that hit only negative keywords (blog/news/legal/etc.)."""
        if u == home:
            return False
        pos, neg = _score_components(u)
        return neg > 0 and pos == 0

    ranked = sorted(
        collected,
        key=lambda u: (-_score_components(u)[0], _score_components(u)[1], len(u)),
    )
    out: list[str] = []
    seen_out: set[str] = set()

    if include_base_in_output and home not in seen_out and _is_html_url(home):
        out.append(home)
        seen_out.add(home)

    for u in ranked:
        u = u.rstrip("/")
        if u in seen_out or not _is_html_url(u):
            continue
        if not _same_site(u, base_netloc):
            continue
        if _is_pure_noise(u):
            continue
        out.append(u)
        seen_out.add(u)
        if len(out) >= max_urls:
            break

    return out[:max_urls]


def discover_program_and_aux_urls(
    base_url: str,
    *,
    max_urls: int = 40,
    seed_depth: int = 2,
    headless: bool = True,
    loader_kwargs: dict | None = None,
    max_aux: int = 2,
) -> tuple[list[str], list[str]]:
    """Run discovery and partition the result into (program_urls, aux_urls).

    ``aux_urls`` is a small set of about/contact/location pages (at most
    ``max_aux``) suitable for the provider-profile extraction prompt. They are
    REMOVED from the returned ``program_urls`` so the program prompt is not
    diluted by non-program content. The base URL is always retained in
    ``program_urls``.
    """
    discovered = discover_program_urls(
        base_url,
        max_urls=max_urls + max_aux,
        seed_depth=seed_depth,
        headless=headless,
        loader_kwargs=loader_kwargs,
    )

    base_canonical = _canonical_base(base_url).rstrip("/")

    aux: list[str] = []
    program: list[str] = []
    for u in discovered:
        if len(aux) < max_aux and u != base_canonical and _is_aux_url(u):
            aux.append(u)
        else:
            program.append(u)

    return program[:max_urls], aux


def iter_unique(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        c = u.split("#", 1)[0].rstrip("/")
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
