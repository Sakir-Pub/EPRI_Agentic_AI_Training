"""
langchain_agent_quickstart.py
-----------------------------
LangChain Quickstart Agent with calculator, Tavily search, file summarizer, and log file analyzer tools.

Author: [Your Name]
License: MIT
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_openai import OpenAI as LangOpenAI  # ✅ Updated to use langchain-openai
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
def tavily_search_tool(query: str) -> str:
    """Use Tavily API to perform a web search and return the top result."""
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
        if "results" in data and data["results"]:
            return data["results"][0].get("content", "No content found.")
        else:
            return "No results returned by Tavily."
    except Exception as e:
        return f"Search error: {e}"

@tool
def summarize_file(filepath: str) -> str:
    """Summarize the contents of a local .txt file."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
        if not content.strip():
            return "File is empty."
        return f"Summary of {filepath}:\n{content[:500]}{'...' if len(content) > 500 else ''}"
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def analyze_log_file(filepath: str) -> str:
    """Analyze a log file and summarize steps, errors, and final answer."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        step_count = sum(1 for line in lines if line.strip().startswith("Step "))
        final_answers = [line for line in lines if "Final Answer:" in line]
        errors = [line for line in lines if "Error" in line or "Traceback" in line]

        summary = [
            f"Log Analysis of {filepath}:",
            f"- Total Steps: {step_count}",
            f"- Final Answers Found: {len(final_answers)}",
            f"- Errors Detected: {len(errors)}"
        ]
        if final_answers:
            summary.append(f"- Last Final Answer: {final_answers[-1].strip()}")
        return "\n".join(summary)
    except Exception as e:
        return f"Log file analysis failed: {e}"

# Step 3: Register tools
tools = [
    Tool.from_function(
        func=simple_calculator,
        name="Calculator",
        description="Evaluate a simple math expression like '3 + 5 * 2'."
    ),
    Tool.from_function(
        func=tavily_search_tool,
        name="Search",
        description="Use Tavily to search the web for a given query string."
    ),
    Tool.from_function(
        func=summarize_file,
        name="Summarizer",
        description="Summarize the contents of a local text file given its full file path."
    ),
    Tool.from_function(
        func=analyze_log_file,
        name="LogAnalyzer",
        description="Analyze a .log file to summarize steps, errors, and final answers."
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
