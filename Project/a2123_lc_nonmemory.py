"""
A2123_non_memory_agent_baseline.py
----------------------------------
LangChain agent WITHOUT memory for comparison.

Author: [Your Name]
License: MIT
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI as LangOpenAI
from dotenv import load_dotenv
import os
import requests
import math

# Step 1: Load environment
load_dotenv()

# Step 2: Define tools

def simple_calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 3 * 5'"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def tavily_search_tool(query: str) -> str:
    """Perform a web search using the Tavily API and return the top result."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "TAVILY_API_KEY not set in environment."
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": 1},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data["results"][0].get("content", "No content returned.") if data.get("results") else "No results."
    except Exception as e:
        return f"Search error: {e}"

# Step 3: Register tools
TOOLS = [
    Tool.from_function(simple_calculator, name="Calculator", description=simple_calculator.__doc__),
    Tool.from_function(tavily_search_tool, name="Search", description=tavily_search_tool.__doc__)
]

# Step 4: Initialize agent WITHOUT memory
llm = LangOpenAI(temperature=0)
agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 5: Stateless loop
if __name__ == "__main__":
    print("🧪 Stateless Agent ready. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = agent.run(query)
        print("\n🧠 Final Answer:", result)
        print("-" * 50)
