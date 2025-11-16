import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from blogagentic.graphs.graph_builder import GraphBuilder
from blogagentic.llms.groq_llm import GroqLLM


app = FastAPI()

# Allow Streamlit (default localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/blogs")
async def create_blogs(request: Request):
    """
    Request JSON:
    {
      "topic": "AI agents with LangGraph",
      "language": "french"   # optional
    }
    """
    data = await request.json()
    topic = data.get("topic", "").strip()
    language = (data.get("language") or "").strip().lower()

    groqllm = GroqLLM()
    llm = groqllm.get_llm()
    graph_builder = GraphBuilder(llm)

    if topic and language and language != "english":
        graph = graph_builder.setup_graph(usecase="language")
        state = graph.invoke(
            {"topic": topic, "current_language": language}
        )
    elif topic:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})
    else:
        return {"error": "Topic is required."}

    return {"data": state}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
