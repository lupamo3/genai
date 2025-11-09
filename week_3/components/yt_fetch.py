from typing import Optional, Sequence, Tuple
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)
from pytube import extract

# yt-dlp fallback
import re, requests, html
from urllib.parse import urljoin
from yt_dlp import YoutubeDL
import xml.etree.ElementTree as ET

def extract_video_id(url: str) -> Optional[str]:
    try:
        return extract.video_id(url)
    except Exception:
        return None

def _clean_vtt_to_text(vtt_text: str) -> str:
    lines = []
    for line in vtt_text.splitlines():
        if not line.strip(): 
            continue
        if line.startswith("WEBVTT"):
            continue
        # strip timestamps and cue numbers
        if re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}", line):
            continue
        if re.match(r"^\d+$", line.strip()):
            continue
        lines.append(line.strip())
    return " ".join(lines)

def _parse_srv3_xml(xml_text: str) -> str:
    """YouTube SRV3 XML -> plain text"""
    try:
        root = ET.fromstring(xml_text)
        texts = []
        for t in root.iter():
            if t.tag.lower().endswith('text') and (t.text or "").strip():
                texts.append(html.unescape(t.text.strip()))
        return " ".join(texts)
    except Exception:
        return ""

def _fetch_vtt_or_playlist(url: str) -> str:
    """
    Download a VTT file, or if it's an M3U8 playlist of VTT chunks,
    follow segments and join.
    """
    r = requests.get(url, timeout=20)
    if not r.ok or not r.text:
        return ""
    text = r.text

    # M3U8 playlist?
    if text.lstrip().startswith("#EXTM3U"):
        # collect segment URLs and fetch them
        vtt_all = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seg_url = urljoin(url, line)
            try:
                seg = requests.get(seg_url, timeout=20)
                if seg.ok and seg.text:
                    vtt_all.append(seg.text)
            except Exception:
                continue
        return _clean_vtt_to_text("\n".join(vtt_all))
    else:
        # assume it is VTT already
        return _clean_vtt_to_text(text)

def _try_ytdlp_captions(video_url: str, langs: Sequence[str]) -> Tuple[str, str]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as e:
        return "", f"yt-dlp extract_info failed: {e}"

    # helper to try a list of tracks (yt-dlp format)
    def try_tracks(tracks_dict, label: str) -> Tuple[str, str]:
        if not tracks_dict:
            return "", ""
        # Prefer entries that look like VTT first; else SRV3; else anything
        entries = []
        for lang in langs:
            entries += [(lang, e) for e in tracks_dict.get(lang, [])]
        # sort by extension preference
        def rank(e):
            ext = (e.get("ext") or "").lower()
            if ext == "vtt": return 0
            if ext == "srv3": return 1
            if ext == "ttml": return 2
            if ext == "json3": return 3
            return 4
        entries.sort(key=lambda pair: rank(pair[1]))

        for lang, entry in entries:
            u = entry.get("url")
            if not u:
                continue
            ext = (entry.get("ext") or "").lower()
            try:
                if ext == "vtt":
                    txt = _fetch_vtt_or_playlist(u)
                    if txt: return txt, f"yt-dlp OK ({label}:{lang}:vtt)"
                elif ext in ("m3u8", ""):
                    # sometimes ext is blank but URL yields m3u8 -> handle in _fetch_vtt_or_playlist
                    txt = _fetch_vtt_or_playlist(u)
                    if txt: return txt, f"yt-dlp OK ({label}:{lang}:m3u8)"
                elif ext == "srv3":
                    resp = requests.get(u, timeout=20)
                    if resp.ok and resp.text:
                        txt = _parse_srv3_xml(resp.text)
                        if txt: return txt, f"yt-dlp OK ({label}:{lang}:srv3)"
                else:
                    # last resort: try fetch and hope it's readable VTT/XML
                    resp = requests.get(u, timeout=20)
                    if resp.ok and resp.text:
                        # guess format
                        body = resp.text
                        if body.lstrip().startswith("#EXTM3U"):
                            txt = _fetch_vtt_or_playlist(u)
                        elif "<text" in body and "</text>" in body:
                            txt = _parse_srv3_xml(body)
                        else:
                            txt = _clean_vtt_to_text(body)
                        if txt: return txt, f"yt-dlp OK ({label}:{lang}:{ext or 'unknown'})"
            except Exception:
                continue
        return "", ""

    # 1) human subtitles first
    txt, reason = try_tracks(info.get("subtitles") or {}, "subtitles")
    if txt:
        return txt, reason

    # 2) auto captions
    txt, reason = try_tracks(info.get("automatic_captions") or {}, "automatic_captions")
    if txt:
        return txt, reason

    return "", "yt-dlp found no usable tracks"

def _get_transcript_via_yta(vid: str, langs: Sequence[str]) -> Tuple[str, str]:
    """
    Use ONLY list_transcripts()/fetch(), never get_transcript.
    Works across older versions too.
    """
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(vid)
    except Exception as e:
        return "", f"yta list_transcripts failed: {e}"

    # Prefer manually-created (exact langs in order)
    for lang in langs:
        try:
            t = tlist.find_manually_created_transcript([lang])
            return " ".join(c["text"] for c in t.fetch()), f"yta manual OK ({lang})"
        except Exception:
            pass

    # Then auto-generated
    try:
        t = tlist.find_generated_transcript(list(langs))
        return " ".join(c["text"] for c in t.fetch()), "yta auto OK"
    except Exception:
        pass

    # Lastly, any available transcript
    for tr in tlist:
        try:
            return " ".join(c["text"] for c in tr.fetch()), f"yta any OK ({tr.language_code})"
        except Exception:
            continue

    return "", "yta no usable tracks"

def fetch_transcript(url: str, lang_priority: Sequence[str] = ("en", "en-US", "en-GB")) -> str:
    text, _ = fetch_transcript_debug(url, lang_priority)
    return text

def fetch_transcript_debug(url: str, lang_priority: Sequence[str] = ("en", "en-US", "en-GB")) -> Tuple[str, str]:
    vid = extract_video_id(url)
    if not vid:
        return "", "couldn't parse video id"

    # 1) youtube-transcript-api via list_transcripts
    text, reason = _get_transcript_via_yta(vid, lang_priority)
    if text:
        return text, reason

    # 2) yt-dlp fallback
    text, reason2 = _try_ytdlp_captions(url, lang_priority)
    if text:
        return text, reason2

    return "", f"{reason}; {reason2}"
