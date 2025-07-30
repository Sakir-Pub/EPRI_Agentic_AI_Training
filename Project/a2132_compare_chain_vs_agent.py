"""
A2132_summary_to_email_agent.py
----------------------------------
LangChain agent version of summary-to-email task.
Performs reasoning in a single agent step with tool use.

Author: [Your Name]
License: MIT
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI
from langchain.tools import tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)

# Define tools
@tool
def summarize_text(text: str) -> str:
    """Summarize a document into a few sentences."""
    return f"Summary: This is a placeholder summary of the input text: '{text[:100]}'..."

@tool
def generate_email(summary: str) -> str:
    """Turn a summary into a short, professional email."""
    return f"Subject: Summary Email\n\nDear colleague,\n\nHere is a summary of the document:\n{summary}\n\nBest regards."

# Register tools
tools = [
    Tool.from_function(summarize_text, name="Summarize", description=summarize_text.__doc__),
    Tool.from_function(generate_email, name="EmailGenerator", description=generate_email.__doc__)
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run interactively
if __name__ == "__main__":
    print("🤖 Agent (Summary → Email) ready. Type 'exit' to quit.\n")
    while True:
        doc = input("Paste your document text: ")
        if doc.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = agent.run(doc)
        print("\n📧 Final Output:\n", result)
        print("-" * 50)
