import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# Agent State
# ==========================================

class AgentState(TypedDict):
    query: str
    route: str
    tool_result: str
    final_answer: str
    attempts: int


# ==========================================
# Router / Agent Decision
# ==========================================

def router(state: AgentState):

    query = state["query"].lower()

    if "calculate" in query:
        route = "calculator"

    elif "keyword" in query:
        route = "keywords"

    else:
        route = "general"

    return {
        "route": route
    }


# ==========================================
# Calculator Tool
# ==========================================

def calculator(state: AgentState):

    query = state["query"]

    expression = query.lower().replace(
        "calculate", ""
    ).strip()

    for attempt in range(2):

        try:

            # Allow only basic mathematical characters
            if not re.fullmatch(
                r"[0-9+\-*/().\s]+",
                expression
            ):
                raise ValueError("Invalid mathematical expression")

            result = eval(expression, {"__builtins__": {}}, {})

            return {
                "tool_result": str(result),
                "attempts": attempt + 1
            }

        except Exception:

            print(
                f"Calculator attempt {attempt + 1} failed."
            )

    return {
        "tool_result": "Unable to calculate the expression.",
        "attempts": 2
    }


# ==========================================
# Keyword Extraction Tool
# ==========================================

def keyword_extractor(state: AgentState):

    text = state["query"]

    text = text.lower().replace(
        "extract keywords",
        ""
    ).strip()

    stop_words = {
        "the", "is", "a", "an", "and",
        "or", "of", "to", "in", "on",
        "for", "with", "this", "that",
        "from", "are", "was", "it"
    }

    words = text.split()

    keywords = []

    for word in words:

        word = word.strip(".,!?")

        if (
            word not in stop_words
            and len(word) > 3
        ):
            keywords.append(word)

    return {
        "tool_result": ", ".join(keywords)
    }


# ==========================================
# General LLM Tool
# ==========================================

def general_tool(state: AgentState):

    response = llm.invoke(
        state["query"]
    )

    return {
        "tool_result": response.content
    }


# ==========================================
# Final Agent Response
# ==========================================

def final_response(state: AgentState):

    route = state["route"]
    tool_result = state["tool_result"]

    if route == "calculator":

        answer = f"The calculated result is: {tool_result}"

    elif route == "keywords":

        answer = f"The extracted keywords are: {tool_result}"

    else:

        answer = tool_result

    return {
        "final_answer": answer
    }


# ==========================================
# Create Agentic Graph
# ==========================================

graph = StateGraph(AgentState)


# Nodes
graph.add_node(
    "router",
    router
)

graph.add_node(
    "calculator",
    calculator
)

graph.add_node(
    "keywords",
    keyword_extractor
)

graph.add_node(
    "general",
    general_tool
)

graph.add_node(
    "final_response",
    final_response
)


# ==========================================
# Edges
# ==========================================

graph.add_edge(
    START,
    "router"
)


# Conditional Routing

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "calculator": "calculator",
        "keywords": "keywords",
        "general": "general"
    }
)


# Tool → Final Response

graph.add_edge(
    "calculator",
    "final_response"
)

graph.add_edge(
    "keywords",
    "final_response"
)

graph.add_edge(
    "general",
    "final_response"
)


# Final → END

graph.add_edge(
    "final_response",
    END
)


# Compile Graph

app = graph.compile()


# ==========================================
# Run Agent
# ==========================================

print("=" * 60)
print("        AGENTIC AI PIPELINE")
print("=" * 60)

print("\nAvailable capabilities:")
print("1. Calculator")
print("2. Keyword Extraction")
print("3. General AI Questions")

print("\nType 'exit' to stop.\n")


while True:

    query = input("Enter your query: ")

    if query.lower() == "exit":
        print("\nAgent stopped.")
        break

    initial_state = {
        "query": query,
        "route": "",
        "tool_result": "",
        "final_answer": "",
        "attempts": 0
    }

    result = app.invoke(
        initial_state
    )

    print("\n" + "=" * 60)

    print("AGENT DECISION:")
    print(result["route"])

    print("\nTOOL RESULT:")
    print(result["tool_result"])

    print("\nFINAL ANSWER:")
    print(result["final_answer"])

    print("=" * 60 + "\n")