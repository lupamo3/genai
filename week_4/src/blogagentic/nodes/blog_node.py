from langchain_core.messages import HumanMessage
from blogagentic.states.blog_state import BlogState
from blogagentic.tools.web_research import WebResearcher


class BlogNode:
    """
    Node implementations for blog generation.
    """

    def __init__(self, llm):
        self.llm = llm
        self.researcher = None

    def web_research(self, state: BlogState) -> BlogState:
        """
        Use Tavily to collect web context about the topic
        and attach it to the state as `research`.
        """
        topic = state.get("topic")
        print("🔎 Tavily request for topic:", topic)
        if not topic:
            return state

        if self.researcher is None:
            self.researcher = WebResearcher()

        research_text = self.researcher.research_topic(topic)
        return {"research": research_text}

    def title_creation(self, state: BlogState) -> BlogState:
        """
        Create a title for the blog given a topic.
        """
        topic = state.get("topic")
        if not topic:
            return state

        prompt = """
You are an expert blog content writer. Use Markdown formatting.
Generate a single, creative, SEO-friendly blog title for the topic below.

Topic: {topic}
"""
        system_message = prompt.format(topic=topic)
        response = self.llm.invoke(system_message)

        return {
            "blog": {
                "title": response.content,
                # content will be filled in by the next node
            }
        }

    def content_generation(self, state: BlogState) -> BlogState:
        """
        Generate full blog content for the given topic and title.
        """
        topic = state.get("topic")
        blog = state.get("blog", {})
        research = state.get("research", "")
        if not topic:
            return state

        system_prompt = """
You are an expert blog writer. Use Markdown formatting.

You are writing an article for the topic: {topic}.

You are also given some web research notes (from a search engine).
Use them for facts, structure, and recent information,
but DO NOT copy them verbatim. Rephrase and synthesize.

WEB RESEARCH NOTES:
{research}

Write a detailed blog article:
- Start with a short hook/introduction
- Use headings and subheadings
- Use bullet points where helpful
- End with a short conclusion
"""
        system_message = system_prompt.format(
            topic=topic, 
            research=research or "No research notes available."
        )
        response = self.llm.invoke(system_message)

        return {
            "blog": {
                "title": blog.get("title", ""),
                "content": response.content,
            }
        }

    def translation(self, state: BlogState) -> BlogState:
        """
        Translate the content to the specified language.
        """
        current_language = state.get("current_language")
        blog = state.get("blog", {})
        content = blog.get("content", "")

        if not current_language or not content:
            return state

        translation_prompt = """
Translate the following blog content into {current_language}.
- Maintain the original tone, style, and Markdown formatting.

CONTENT:
{blog_content}
"""
        message = HumanMessage(
            translation_prompt.format(
                current_language=current_language,
                blog_content=content,
            )
        )
        response = self.llm.invoke([message])

        return {
            "blog": {
                "title": blog.get("title", ""),
                "content": response.content,
            },
            "current_language": current_language,
        }

    def route(self, state: BlogState) -> BlogState:
        """
        Simple router node: just returns the current_language.
        """
        return {"current_language": state.get("current_language", "")}

    def route_decision(self, state: BlogState) -> str:
        """
        Decide where to go next based on current_language.
        """
        lang = (state.get("current_language") or "").lower()
        if lang == "kiswahili":
            return "kiswahili"
        if lang == "spanish":
            return "spanish"
        # default: no translation
        return "end"
