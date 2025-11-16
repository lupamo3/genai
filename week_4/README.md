# Agentic AI Blog Generation System

This project implements an **Agentic Blog Generator** using: -
**LangGraph** for workflow orchestration\
- **Groq (Llama 3.1 8B)** as the LLM provider\
- **Tavily** for web research\
- **FastAPI** backend\
- **Streamlit** frontend\
- **Poetry** for dependency management\
- **Python 3.12**

------------------------------------------------------------------------

## 🚀 Features

-   Topic‑based blog generation\
-   Optional language translation (Kiswahili, )\
-   Tavily‑powered research injected into the workflow\
-   End‑to‑end agentic pipeline (research → title → content →
    translate)\
-   Clean UI via Streamlit

------------------------------------------------------------------------

## 🧱 Project Structure

    week_4/
    ├─ app.py
    ├─ streamlit_app.py
    ├─ pyproject.toml
    ├─ .env.example
    └─ src/blogagentic/
        ├─ llms/
        ├─ nodes/
        ├─ states/
        ├─ graphs/
        └─ tools/
        

------------------------------------------------------------------------

## 🔧 Setup Instructions

### 1. Clone the repository

``` bash
git clone https://github.com/lupamo3/genai
cd week_4
```

### 2. Create your `.env`

Copy the example file:

``` bash
cp .env.example .env
```

Fill in:

    GROQ_API_KEY=your_key
    TAVILY_API_KEY=your_key
    BLOG_API_URL=http://localhost:8000/blogs

### 3. Install dependencies with Poetry

``` bash
poetry install
```

------------------------------------------------------------------------

## ▶️ Running the Application

### Start the backend (FastAPI)

``` bash
poetry run uvicorn app:app --reload
```

### Start the frontend (Streamlit)

``` bash
poetry run streamlit run streamlit_app.py
```

Open Streamlit UI at:

    http://localhost:8501

------------------------------------------------------------------------

## 🧪 Testing the API Directly

    POST /blogs
    Content-Type: application/json

    {
      "topic": "AI agents with LangGraph",
      "language": "spanish"
    }

------------------------------------------------------------------------

## ✔️ Output

The system returns: - Blog Title\
- Blog Content (Markdown)\
- Optional Translated Content

------------------------------------------------------------------------

## 📄 License

This project is for educational use under the Andela GenAI Bootcamp Week
4 assignment.
