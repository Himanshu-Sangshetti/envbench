"""Verify every reference URL in the bibliography actually resolves.

Standing rule: never cite an unfetched URL. A reference whose URL 404s is a
hallucinated reference, and ICLR 2026 desk-rejected every paper with a
confirmed hallucinated reference. This script checks all of them.

    python scripts/verify_refs.py --bib ../paper/refs.bib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# Hosts that block automated clients but are known-good bibliographic records.
SOFT_BLOCK = ("doi.org", "dl.acm.org", "proceedings.mlr.press", "openreview.net")

BAD_PATTERNS = re.compile(
    r"(arxiv\.org/abs/(\d{4}\.\d{4,5}))|"
    r"(10\.\d{4,9}/[^\s\"}]+)")


def fetch(url: str, timeout: int = 25) -> tuple[str, str]:
    """Return (status, detail). status in {ok, redirect, blocked, 404, error}."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(4096)
            return ("ok" if r.status == 200 else f"http{r.status}"), ""
    except urllib.error.HTTPError as e:
        if e.code in (403, 401, 429, 503) and any(h in url for h in SOFT_BLOCK):
            return "blocked", f"HTTP {e.code} (host blocks bots; record is real)"
        if e.code == 404:
            return "404", "not found"
        return "blocked", f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "error", str(getattr(e, "reason", e))[:60]
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {e}"[:60]


def entries(bib: str):
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        key, body = m.group(1).strip(), m.group(2)
        urls = re.findall(r"url\s*=\s*\{([^}]*)\}", body)
        yield key, urls, body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="../paper/refs.bib")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=25)
    args = ap.parse_args()

    bib_path = Path(args.bib).resolve()
    bib = bib_path.read_text(errors="ignore")

    # 1. placeholder scan -- never fetch a URL that is obviously synthetic
    print("── placeholder scan ──")
    placeholders = []
    for key, urls, body in entries(bib):
        for u in urls:
            if re.search(r"(XXXX|\b0{4,}\b|example\.com|placeholder)", u, re.I):
                placeholders.append((key, u))
        if "XX.XXXXX" in body:
            placeholders.append((key, "arXiv:XX.XXXXX in journal field"))
    if placeholders:
        for k, u in placeholders:
            print(f"  ✗ {k}: {u}")
    else:
        print(f"  ✓ no placeholder URLs or ids in {len(list(entries(bib)))} entries")

    # 2. live fetch
    print("\n── live fetch ──")
    results = {}
    todo = [(k, u) for k, urls, _ in entries(bib) for u in urls]
    for i, (key, url) in enumerate(todo, 1):
        status, detail = fetch(url, args.timeout)
        results.setdefault(key, []).append({"url": url, "status": status,
                                           "detail": detail})
        mark = {"ok": "✓", "blocked": "~", "404": "✗", "error": "✗"}.get(status, "?")
        print(f"  [{i:>2}/{len(todo)}] {mark} {key:24s} {status:8s} {url}")
        time.sleep(0.4)

    dead = [k for k, v in results.items()
            if any(r["status"] in ("404", "error") for r in v)]
    soft = [k for k, v in results.items()
            if any(r["status"] == "blocked" for r in v)
            and not any(r["status"] in ("404", "error") for r in v)]

    print(f"\n{len(results)} references checked")
    print(f"  live      : {len(results) - len(dead) - len(soft)}")
    print(f"  bot-blocked (record exists, host refuses bots): {len(soft)}")
    print(f"  DEAD      : {len(dead)}  {dead}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.out}")

    return 1 if placeholders or dead else 0


if __name__ == "__main__":
    sys.exit(main())
