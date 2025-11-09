import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MIN_WORDS = 80  # require at least this many words to consider it extracted


def _summarize(text: str, n=1200) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:n] + ("..." if len(text) > n else "")


def fetch_article_text_debug(url: str):
    """
    Try multiple extraction strategies and return (text, debug_info).
    """
    debug = {"url": url, "steps": []}
    text = ""

    # 1) Try Trafilatura fetch_url → extract
    try:
        downloaded = trafilatura.fetch_url(url)
        debug["steps"].append(
            {"step": "trafilatura.fetch_url", "ok": bool(downloaded), "len": len(downloaded or "")}
        )
        if downloaded:
            t = trafilatura.extract(downloaded, favor_recall=True)
            debug["steps"].append(
                {"step": "trafilatura.extract(downloaded)", "ok": bool(t), "words": len((t or "").split())}
            )
            if t and len(t.split()) >= MIN_WORDS:
                return t, debug
    except Exception as e:
        debug["steps"].append({"step": "trafilatura.fetch/extract EXC", "error": str(e)})

    # 2) Raw GET → Trafilatura extract
    html = ""
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=20)
        r.raise_for_status()
        html = r.text
        debug["steps"].append({"step": "requests.get", "status": r.status_code, "len": len(html)})
        t = trafilatura.extract(html, favor_recall=True)
        debug["steps"].append(
            {"step": "trafilatura.extract(html)", "ok": bool(t), "words": len((t or "").split())}
        )
        if t and len(t.split()) >= MIN_WORDS:
            return t, debug
    except Exception as e:
        debug["steps"].append({"step": "requests.get EXC", "error": str(e)})

    # 3) Readability (fallback)
    try:
        if not html:
            r = requests.get(url, headers=USER_AGENT, timeout=20)
            r.raise_for_status()
            html = r.text
            debug["steps"].append({"step": "requests.get (fallback)", "status": r.status_code, "len": len(html)})

        doc = Document(html)
        title = doc.short_title() or ""
        body_html = doc.summary()
        soup = BeautifulSoup(body_html, "lxml")
        t = soup.get_text("\n", strip=True)
        if title and title not in t[:200]:
            t = f"{title}\n\n{t}"
        debug["steps"].append({"step": "readability+bs4", "ok": bool(t), "words": len((t or '').split())})
        if t and len(t.split()) >= MIN_WORDS:
            return t, debug
    except Exception as e:
        debug["steps"].append({"step": "readability EXC", "error": str(e)})

    # 4) Try simple AMP variant (some sites expose cleaner pages)
    try:
        amp_url = url.rstrip("/") + "/amp"
        r = requests.get(amp_url, headers=USER_AGENT, timeout=20)
        if r.ok:
            t = trafilatura.extract(r.text, favor_recall=True)
            debug["steps"].append(
                {"step": "AMP trafilatura.extract", "ok": bool(t), "words": len((t or '').split()), "amp_url": amp_url}
            )
            if t and len(t.split()) >= MIN_WORDS:
                return t, debug
        else:
            debug["steps"].append({"step": "AMP request", "status": r.status_code, "amp_url": amp_url})
    except Exception as e:
        debug["steps"].append({"step": "AMP EXC", "error": str(e)})

    return "", debug


def fetch_article_text(url: str) -> str:
    text, _ = fetch_article_text_debug(url)
    return text


def clean_summary_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    return text
