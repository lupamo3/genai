import os
from dotenv import load_dotenv
from tavily import TavilyClient


class WebResearcher:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in environment.")

        self.client = TavilyClient(api_key=api_key)

    def research_topic(self, topic: str, max_results: int = 5) -> str:
        """
        Use Tavily to fetch web results and return a compact summary text
        that we can feed into the LLM.
        """
        resp = self.client.search(
            query=topic,
            max_results=max_results,
            search_depth="basic",          
            include_answer=True,
            include_raw_content=False,
        )

        # resp["answer"] is Tavily's synthesized summary (if include_answer=True)
        answer = resp.get("answer") or ""

        # Build a short text blob with answer + top sources
        snippets = []
        if answer:
            snippets.append(f"High-level summary:\n{answer}\n")

        sources = resp.get("results", [])[:max_results]
        for idx, r in enumerate(sources, start=1):
            title = r.get("title") or "Untitled"
            snippet = r.get("content") or ""
            snippets.append(f"Source {idx}: {title}\n{snippet}\n")

        return "\n".join(snippets)
