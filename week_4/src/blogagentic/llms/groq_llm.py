from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


class GroqLLM:
    def __init__(self) -> None:
        # Load environment variables from .env
        load_dotenv()

    def get_llm(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")

        # Optional: you can also set LangSmith for tracing if you want
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key

        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
        )
        return llm
