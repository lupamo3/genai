# streamlit_app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Auto-load environment variables from .env
load_dotenv()

API_URL = os.getenv("BLOG_API_URL", "http://localhost:8000/blogs")

def main():
    st.set_page_config(page_title="📝 Agentic AI Blog Generator", layout="wide")
    st.title("📝 Agentic AI Blog Generation System")

    with st.sidebar:
        st.header("Configuration (Auto-loaded)")
        st.success("🔐 API keys loaded from .env")

        language = st.selectbox(
            "Output language",
            ["English (no translation)", "Kiswahili", "Spanish"],
            index=0,
        )

    topic = st.text_input("Blog topic", placeholder="e.g. Agentic AI with LangGraph")

    if st.button("Generate blog"):
        if not topic.strip():
            st.warning("Please enter a topic.")
            st.stop()

        lang_value = ""
        if "kiswahili" in language.lower():
            lang_value = "kiswahili"
        elif "spanish" in language.lower():
            lang_value = "spanish"
        else:
            lang_value = "english"

        with st.spinner("Generating blog..."):
            payload = {"topic": topic, "language": lang_value}
            resp = requests.post(API_URL, json=payload)

        if resp.status_code != 200:
            st.error(f"API error: {resp.status_code} - {resp.text}")
            st.stop()

        data = resp.json().get("data", {})
        blog = data.get("blog", {})

        title = blog.get("title", "").strip()
        content = blog.get("content", "").strip()

        st.subheader("Generated Title")
        st.write(title)

        st.subheader("Generated Blog")
        st.markdown(content, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
