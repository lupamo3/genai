from typing import TypedDict, NotRequired, Dict


class BlogState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.
    """
    topic: str
    blog: Dict[str, str]        
    current_language: NotRequired[str]
    research: NotRequired[str]
