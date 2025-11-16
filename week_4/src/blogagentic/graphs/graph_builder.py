from langgraph.graph import StateGraph, START, END

from blogagentic.states.blog_state import BlogState
from blogagentic.nodes.blog_node import BlogNode


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm

    def build_topic_graph(self):
        """
        Build a graph to generate blogs based only on topic.
        """
        graph = StateGraph(BlogState)
        blog_node_obj = BlogNode(self.llm)

        # Nodes
        graph.add_node("title_creation", blog_node_obj.title_creation)
        graph.add_node("content_generation", blog_node_obj.content_generation)

        # Edges
        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generation")
        graph.add_edge("content_generation", END)

        return graph

    def build_language_graph(self):
        """
        Build a graph for blog generation with topic and language.
        """
        graph = StateGraph(BlogState)
        blog_node_obj = BlogNode(self.llm)

        # Nodes
        graph.add_node("title_creation", blog_node_obj.title_creation)
        graph.add_node("content_generation", blog_node_obj.content_generation)
        graph.add_node(
            "swahili_translation",
            lambda state: blog_node_obj.translation(
                {**state, "current_language": "kiswahili"}
            ),
        )
        graph.add_node(
            "spanish_translation",
            lambda state: blog_node_obj.translation(
                {**state, "current_language": "spanish"}
            ),
        )
        graph.add_node("route", blog_node_obj.route)

        # Edges
        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generation")
        graph.add_edge("content_generation", "route")

        # Conditional edges from router
        graph.add_conditional_edges(
            "route",
            blog_node_obj.route_decision,
            {
                "kiswahili": "swahili_translation",
                "spanish": "spanish_translation",
                "end": END,
            },
        )

        graph.add_edge("swahili_translation", END)
        graph.add_edge("spanish_translation", END)

        return graph

    def setup_graph(self, usecase: str):
        """
        Compile and return the graph depending on usecase.
        """
        if usecase == "topic":
            graph = self.build_topic_graph()
        elif usecase == "language":
            graph = self.build_language_graph()
        else:
            raise ValueError(f"Unknown usecase: {usecase}")

        return graph.compile()
