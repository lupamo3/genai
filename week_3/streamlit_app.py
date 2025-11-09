import io
import os
import streamlit as st
from llm_client import LLMClient
from components.news_fetch import fetch_article_text_debug, clean_summary_text
from components.yt_fetch import fetch_transcript, fetch_transcript_debug, extract_video_id
from components.rag_utils import chunk_docs, build_chroma, answer_with_rag

from appconfig import env_config as env  # optional, kept for parity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
default_groq = os.getenv("GROQ_API_KEY", "")
default_openai = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Week 3 – Multi-App (News/YouTube/Voice RAG)", page_icon="🧠", layout="wide")

# --- Sidebar: global controls (Week 1 features) ---
st.sidebar.header("Global Configuration")

# Keys: allow override via sidebar
_groq = st.sidebar.text_input("Groq API Key", type="password", value=default_groq)
_openai = st.sidebar.text_input("OpenAI API Key", type="password", value=default_openai)

model = st.sidebar.selectbox(
    "Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
)

chatbot_name = st.sidebar.text_input("Chatbot Name", value="Atlas")
system_prompt = st.sidebar.text_area("System Prompt (optional)")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 256, 4096, 1024, 256)

# Create LLM client (auto-switch)
key = _openai if model.startswith("gpt-") else _groq
try:
    llm = LLMClient(
        model=model,
        api_key=key,
        chatbot_name=chatbot_name,
        default_system_prompt=(system_prompt or None),
    )
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.button("Clear Conversation", on_click=llm.clear, use_container_width=True)
provider = "OpenAI" if model.startswith("gpt-") else "Groq"
st.sidebar.caption(f"Provider: **{provider}** • Model: **{model}**")

st.title("Week 3 Projects – News / YouTube / Voice Assistant RAG")
st.write("Three tools below reusing my Week 1 controls (keys, model, persona, temp, tokens).")

# --- Long-article safety helpers ---
def approx_tokens(s: str) -> int:
    # crude 1 token ≈ 4 chars heuristic
    return max(1, len(s) // 4)

def chunk_text(s: str, max_chars: int = 10000, overlap: int = 800) -> list[str]:
    """Split long text into overlapping chunks to stay under provider limits."""
    if len(s) <= max_chars:
        return [s]
    chunks: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        end = min(i + max_chars, n)
        chunk = s[i:end]
        chunks.append(chunk)
        if end == n:
            break
        i = end - overlap  # step with overlap
        if i < 0:
            i = 0
    return chunks

def summarize_chunk(text: str, style: str) -> str:
    prompt = f"""Summarize the following article chunk in the style: {style}.
- Be concise and factual.
- 5–8 bullets if applicable.
- Do NOT reference “this chunk”; write a standalone summary.

Chunk:
{text}
"""
    return call_llm(prompt)

def synthesize_summaries(part_summaries: list[str], style: str) -> str:
    joined = "\n\n".join(f"- {s}" for s in part_summaries)
    prompt = f"""You are combining multiple partial summaries of one article into a single {style} for an end user.

Instructions:
- Merge and deduplicate ideas.
- Keep it crisp and non-repetitive.
- Output 1–2 short paragraphs for “Executive Summary”, or 7–10 bullets for bullet styles.

Partial summaries:
{joined}

Now produce the final {style}:"""
    return call_llm(prompt)

# --- Utility wrappers ---
def call_llm(prompt: str) -> str:
    return llm.chat(prompt, temperature=temperature, max_tokens=max_tokens)

# -----------------
# 1) News Summarizer
# -----------------
st.header("🗞️ News Summarizer")
col1, col2 = st.columns([2, 1])
with col1:
    url = st.text_input("Article URL", placeholder="https:// ...")
    mode = st.selectbox("Summary Style", ["Bullet Points", "Executive Summary", "Key Takeaways"], index=1)

    if st.button("Summarize Article", type="primary"):
        if not url:
            st.warning("Please paste an article URL.")
        else:
            text, dbg = fetch_article_text_debug(url)
            if not text:
                st.error("Could not fetch/parse the article.")
                with st.expander("Debug details"):
                    st.json(dbg)
            else:
                with st.expander("🔎 Extraction preview", expanded=False):
                    st.write(f"Extracted words: {len(text.split())}")
                    st.code(text[:1200] + ("..." if len(text) > 1200 else ""), language="markdown")

                # --- token-safe path ---
                try:
                    # Heuristic caps tuned for Groq on-demand limits; adjust if needed
                    MAX_CHARS_SINGLE = 12000     # ~3k tokens input
                    MAX_CHARS_CHUNK  = 9000      # ~2.2k tokens per chunk
                    OVERLAP_CHARS    = 600

                    if len(text) <= MAX_CHARS_SINGLE:
                        prompt = f"""Summarize the article below in the style: {mode}.
- Keep it factual and concise.
- Include 5–8 bullets when applicable.

Article:
{text}
"""
                        summary = call_llm(prompt)
                    else:
                        chunks = chunk_text(text, max_chars=MAX_CHARS_CHUNK, overlap=OVERLAP_CHARS)
                        st.caption(f"Long article detected. Using chunked summarization ({len(chunks)} chunks).")
                        part_summaries: list[str] = []
                        for idx, ch in enumerate(chunks, 1):
                            # Use spinner/status to show progress per chunk
                            with st.status(f"Summarizing chunk {idx}/{len(chunks)}...", expanded=False):
                                part = summarize_chunk(ch, mode)
                                part_summaries.append(part)
                        summary = synthesize_summaries(part_summaries, mode)

                except Exception as e:
                    # Last-resort truncation if provider still complains (TPM/length)
                    ST_TRUNC = 8000
                    st.warning(f"Fell back to truncated summarization due to provider limits: {e}")
                    short = text[:ST_TRUNC]
                    prompt = f"""Summarize the article below in the style: {mode}.
- Keep it factual and concise.
- Include 5–8 bullets when applicable.

Article (truncated):
{short}
"""
                    summary = call_llm(prompt)

                st.subheader("Summary")
                st.write(clean_summary_text(summary))

with col2:
    st.markdown("**Tips**")
    st.markdown("- Works best on public, readable articles.\n- Try Executive Summary for 1–2 paragraphs; Bullet Points for lists.")

st.divider()

# --------------------
# 2) YouTube Summarizer
# --------------------
st.header("📺 YouTube Summarizer")
col1, col2 = st.columns([2, 1])
with col1:
    yt = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    style = st.selectbox("Summary Style", ["Timestamps", "Bullet Points", "One-paragraph Recap"], index=0)
    if st.button("Summarize Video", type="primary"):
        if not yt:
            st.warning("Please paste a YouTube URL.")
        else:
            # use debug helper
            transcript, dbg = fetch_transcript_debug(yt)
            if not transcript:
                st.error(f"No transcript found or subtitles disabled. (videoId: {extract_video_id(yt) or 'n/a'})")
                st.info(f"Debug: {dbg}")
            else:
                prompt = f"""Summarize this YouTube transcript in style = {style}.
- If 'Timestamps', group key moments with approximate timestamps every 1–3 minutes.
- If 'Bullet Points', return 7–10 bullets.
- If 'One-paragraph Recap', return 5–7 sentences.

Transcript:
{transcript}
"""
                out = call_llm(prompt)
                st.subheader("Summary")
                st.write(clean_summary_text(out))
with col2:
    st.markdown("**Notes**")
    st.markdown("- Requires public transcript.\n- Auto-captions often available in English.")

st.divider()

# -----------------------
# 3) Voice Assistant RAG
# -----------------------
st.header("🎙️ Voice Assistant RAG")
st.markdown("Upload a few docs, ask by **voice or text**, and get answers grounded in your corpus.")

rag_col1, rag_col2 = st.columns([2, 1])
with rag_col1:
    up = st.file_uploader(
        "Upload knowledge files (txt/pdf/docx supported as text)",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if up:
        raw_texts = []
        sources = []
        for f in up:
            try:
                # Best-effort text read; keep simple for now
                data = f.read()
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = data.decode("latin-1", errors="ignore")
                raw_texts.append(text)
                sources.append(f.name)
            except Exception:
                pass
        docs = chunk_docs(raw_texts, sources=sources)
        vectordb = build_chroma(docs, collection_name="voice_rag")
        st.success(f"Loaded {len(docs)} chunks from {len(up)} files.")

        q = st.text_input("Ask a question (or use voice input below)")
        if st.button("Ask RAG") and q:
            ans = answer_with_rag(vectordb, call_llm, q)
            st.subheader("Answer")
            st.write(ans)

with rag_col2:
    st.markdown("**Voice (optional)**")
    st.caption("Upload a short .wav/.mp3 question. If OpenAI key is set, we attempt Whisper STT.")
    audio = st.file_uploader("Upload audio question", type=["wav", "mp3", "m4a"])
    if audio and _openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=_openai)
            audio_bytes = audio.read()
            # Save to temp and send to Whisper
            buf = io.BytesIO(audio_bytes)
            buf.name = audio.name
            # The SDK expects a file-like with .name
            st.info("Transcribing...")
            transcript = client.audio.transcriptions.create(model="whisper-1", file=buf)
            query = transcript.text
            st.write(f"**You said:** {query}")
            if 'vectordb' in locals():
                ans = answer_with_rag(vectordb, call_llm, query)
                st.subheader("Answer")
                st.write(ans)
            else:
                st.warning("Please upload knowledge files first, then ask.")
        except Exception as e:
            st.error(f"STT failed: {e}")
    elif audio and not _openai:
        st.warning("OpenAI key required for Whisper transcription.")
