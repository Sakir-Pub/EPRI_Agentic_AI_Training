"""
langchain_agent_quickstart.py
-----------------------------
LangChain Quickstart Agent with calculator and search tools.

Author: [Your Name]
License: MIT
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_openai import OpenAI as LangOpenAI
from dotenv import load_dotenv
import os
import math
import requests

# Step 1: Load environment variables
load_dotenv()

# Step 2: Create basic tools
@tool
def simple_calculator(expression: str) -> str:
    """Evaluate a simple math expression (e.g., '3 + 5 * 2')."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool
def dummy_search(query: str) -> str:
    """Simulate a search response (replace with real API if needed)."""
    return f"Pretend search result for: '{query}'"

# Step 3: Register tools
tools = [
    Tool.from_function(
        func=simple_calculator,
        name="Calculator",
        description="Evaluate a simple math expression like '3 + 5 * 2'."
    ),
    Tool.from_function(
        func=dummy_search,
        name="Search",
        description="Simulate a web search for a given query string."
    )
]
# Step 4: Initialize agent
llm = LangOpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 5: Run agent
if __name__ == "__main__":
    query = input("Ask a question: ")
    result = agent.run(query)
    print("\n🧠 Final Answer:", result)
