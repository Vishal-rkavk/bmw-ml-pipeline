"""Step 1: collect BMW press articles into data/raw/ (idempotent).

Discovery uses the newsroom sitemap; metadata comes from each page's
schema.org JSON-LD block and the body from its substantial <p>
paragraphs. The per-article JSON files are the corpus — a URL that was
already collected is never fetched again, and identical texts (sha256)
are dropped. MAX_DOCS is the total corpus target: a run tops up to it,
and a re-run at target is a no-op. Rejected URLs are cached in
_rejected.json so they are not fetched again either.
"""

import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from pipeline import config, metrics

USER_AGENT = "BMW-ML-Pipeline/0.1 (educational take-home assignment)"
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
REJECTED_PATH = config.RAW_DIR / "_rejected.json"
ARTICLE_URL = re.compile(
    rf"^{re.escape(config.BASE_URL)}/global/[a-z0-9-]+/$")


def doc_path(slug: str, sha: str) -> Path:
    """Filesystem-safe, readable, content-unique article path.

    Slugs can exceed the 255-byte filename limit (a 247-char slug
    crashed the first full run), so the slug is truncated and the
    content hash suffix guarantees uniqueness.
    """
    return config.RAW_DIR / f"{slug[:96]}-{sha[:8]}.json"


def fetch(session: requests.Session, url: str) -> str | None:
    """GET with timeout, retries and exponential backoff; None on failure."""
    for attempt in range(config.MAX_RETRIES):
        try:
            response = session.get(url, timeout=config.REQUEST_TIMEOUT_S)
            if response.status_code == 200:
                return response.text
            if response.status_code not in (429, 500, 502, 503):
                return None  # e.g. 404 — retrying will not help
        except requests.RequestException:
            pass  # timeout / connection error — retry after backoff
        time.sleep(2 ** attempt)
    print(f"[warn] giving up on {url} after {config.MAX_RETRIES} attempts")
    return None


def article_urls(sitemap_xml: str) -> list[str]:
    """English article URLs from the sitemap, in sitemap order."""
    root = ET.fromstring(sitemap_xml)
    urls = []
    for loc in root.iter(f"{SITEMAP_NS}loc"):
        url = (loc.text or "").strip()
        if not ARTICLE_URL.match(url):
            continue
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        if slug in config.EXCLUDE_PATHS:
            continue
        urls.append(url)
    return urls


def parse_article(html: str) -> dict | None:
    """Extract title/date/text from an article page.

    Returns None when the page is not a proper press article: missing
    JSON-LD metadata, unparseable metadata, or too little body text.
    Policy decisions (date cutoff, size threshold beyond the minimum)
    are made by the caller, not here.
    """
    soup = BeautifulSoup(html, "html.parser")
    ld_script = soup.find("script", type="application/ld+json")
    if ld_script is None or not ld_script.string:
        return None
    try:
        ld = json.loads(ld_script.string)
    except json.JSONDecodeError:
        return None

    title = ld.get("headline") or ""
    date = (ld.get("datePublished") or "")[:10]  # "2026-09-30T..." -> date
    if not title or len(date) != 10:
        return None

    paragraphs = []
    for p in soup.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(text) < 80:
            continue  # captions, labels, button text, ...
        if text.count("http") >= 2:
            continue  # link lists are navigation, not prose
        paragraphs.append(text)
    body = "\n\n".join(paragraphs)
    if len(body) < config.MIN_TEXT_CHARS:
        return None
    return {"title": title.strip(), "date": date, "text": body}


def load_corpus() -> tuple[set[str], set[str], dict[str, str]]:
    """Current corpus state: (collected urls, text hashes, rejected urls)."""
    urls: set[str] = set()
    hashes: set[str] = set()
    for path in config.RAW_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        urls.add(doc["url"])
        hashes.add(doc["sha256"])
    rejected: dict[str, str] = {}
    if REJECTED_PATH.exists():
        rejected = json.loads(REJECTED_PATH.read_text(encoding="utf-8"))
    return urls, hashes, rejected


def _save_rejected(rejected: dict[str, str]) -> None:
    """Persist the rejected-URL cache (_rejected.json)."""
    REJECTED_PATH.write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> int:
    config.ensure_dirs()
    started = time.time()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"[collect] fetching sitemap: {config.SITEMAP_URL}")
    sitemap_xml = fetch(session, config.SITEMAP_URL)
    if sitemap_xml is None:
        print("[collect][FAIL] could not fetch the sitemap — aborting")
        return 1
    candidates = article_urls(sitemap_xml)
    print(f"[collect] {len(candidates)} candidate article URLs "
          f"(target: {config.MAX_DOCS}, min date: {config.MIN_DATE})")

    known_urls, known_hashes, rejected = load_corpus()
    new_hashes: set[str] = set()
    counts = {"ok": 0, "duplicate": 0, "error": 0, "skipped": 0}

    # Top up the corpus to MAX_DOCS total (not MAX_DOCS new per run):
    # that is what makes re-runs at target exact no-ops.
    target_new = max(0, config.MAX_DOCS - len(known_urls))
    if target_new == 0:
        print(f"[collect] corpus already at target "
              f"({len(known_urls)} docs) — nothing to do")
    else:
        print(f"[collect] topping up: {target_new} more docs needed")

    try:
        for i, url in enumerate(candidates, start=1):
            if counts["ok"] >= target_new:
                break
            if url in known_urls:
                counts["duplicate"] += 1
                continue
            if url in rejected:
                counts["skipped"] += 1
                continue

            # Per-article guard: one strange page must never kill the
            # whole run. Errors are counted, logged loudly, and skipped.
            try:
                html = fetch(session, url)
                time.sleep(config.REQUEST_DELAY_S)
                if html is None:
                    counts["error"] += 1
                    continue

                article = parse_article(html)
                if article is None:
                    rejected[url] = "not_article"
                    counts["skipped"] += 1
                    continue
                if article["date"] < config.MIN_DATE:
                    rejected[url] = "too_old"
                    counts["skipped"] += 1
                    continue

                sha = hashlib.sha256(
                    article["text"].encode("utf-8")).hexdigest()
                if sha in known_hashes or sha in new_hashes:
                    rejected[url] = "duplicate_text"
                    counts["duplicate"] += 1
                    continue

                slug = url.rstrip("/").rsplit("/", 1)[-1]
                doc = {
                    "url": url,
                    "title": article["title"],
                    "date": article["date"],
                    "text": article["text"],
                    "sha256": sha,
                    "fetched_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                }
                doc_path(slug, sha).write_text(
                    json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
                known_urls.add(url)
                new_hashes.add(sha)
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001 - keep collecting
                print(f"[warn] error processing {url}: {exc!r}")
                counts["error"] += 1
                continue

            if counts["ok"] % 25 == 0 or i % 100 == 0:
                print(f"[collect] {i}/{len(candidates)} seen | "
                      f"new={counts['ok']} dup={counts['duplicate']} "
                      f"err={counts['error']} skip={counts['skipped']}")
    except KeyboardInterrupt:
        print("\n[collect] interrupted — keeping everything collected so far")
        _save_rejected(rejected)

    _save_rejected(rejected)
    duration = time.time() - started
    size = sum(1 for p in config.RAW_DIR.glob("*.json")
               if not p.name.startswith("_"))
    print(f"[collect] done in {duration:.0f}s | new={counts['ok']} "
          f"dup={counts['duplicate']} err={counts['error']} "
          f"skip={counts['skipped']} | corpus={size} docs")

    metrics.push(
        "bmw-collect",
        values={
            "bmw_collect_new_docs": counts["ok"],
            "bmw_collect_duration_seconds": round(duration, 1),
            "bmw_corpus_docs": size,
        },
        series={"bmw_collect_docs": ("status", counts)},
    )
    return 0 if size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
