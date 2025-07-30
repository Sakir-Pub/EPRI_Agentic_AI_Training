"""
langchain_agent_memory_enabled.py
----------------------------------
LangChain Agent with calculator, Tavily search, file summarizer, log analyzer, short-term memory, and custom PromptTemplate + Tool Routing.

Author: [Your Name]
License: MIT
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI as LangOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import math
import requests

# Step 1: Load environment variables
load_dotenv()

# Step 2: Create base tools
@tool
def simple_calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 3 * 5'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool
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
        if "results" in data and data["results"]:
            return data["results"][0].get("content", "No content found.")
        else:
            return "No results returned by Tavily."
    except Exception as e:
        return f"Search error: {e}"

@tool
def summarize_file(filepath: str) -> str:
    """Summarize the content of a local .txt file given its path."""
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
    """Analyze a .log file and summarize key events and errors."""
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

# Step 3: Tool router based on query keyword triggers
def get_routed_tools(query):
    selected = []
    if any(kw in query.lower() for kw in ["calculate", "add", "multiply", "+", "*", "what is"]):
        selected.append(Tool.from_function(
            func=simple_calculator,
            name="Calculator",
            description=simple_calculator.__doc__ or "No description provided."
        ))
    if any(kw in query.lower() for kw in ["search", "google", "find", "lookup"]):
        selected.append(Tool.from_function(
            func=tavily_search_tool,
            name="Search",
            description=tavily_search_tool.__doc__ or "No description provided."
        ))
    if any(kw in query.lower() for kw in ["summarize", ".txt", "file content"]):
        selected.append(Tool.from_function(
            func=summarize_file,
            name="Summarizer",
            description=summarize_file.__doc__ or "No description provided."
        ))
    if any(kw in query.lower() for kw in ["analyze log", ".log"]):
        selected.append(Tool.from_function(
            func=analyze_log_file,
            name="LogAnalyzer",
            description=analyze_log_file.__doc__ or "No description provided."
        ))
    return selected

# Step 4: Initialize LLM, memory, prompt
llm = LangOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are a helpful and intelligent AI assistant.
You have memory of prior messages in this session stored in 'chat_history'.
Use the memory to answer follow-up questions or recall previous context.
Only use tools when strictly necessary. Default to using memory if applicable.

{chat_history}
User: {input}
Assistant:"""
)

# Step 5: Run interactive loop
if __name__ == "__main__":
    print("🤖 Agent ready. Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break
        tools = get_routed_tools(query)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        result = agent.run(query)
        print("\n🧠 Final Answer:", result)
        print("-" * 50)
