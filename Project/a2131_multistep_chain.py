"""
A2131_summary_to_email_chain.py
----------------------------------
LangChain multi-step chain example:
Step 1: Summarize a document
Step 2: Generate an email from the summary

Author: [Your Name]
License: MIT
"""

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableMap
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)

# Step 1: Summarization Prompt
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Please read the following document and generate a concise summary:

{text}

Summary:
"""
)

# Step 2: Email Generation Prompt (revised for grounding)
email_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
Using only the information in the summary below, write a short professional email explaining the document’s content.

Summary:
{summary}

Email:
"""
)

# Define runnable chain (corrected syntax)
chain = (
    summary_prompt
    | llm
    | (lambda summary: {"summary": summary})
    | email_prompt
    | llm
)

# Run interactively
if __name__ == "__main__":
    print("📄 Multi-Step Chain (Doc → Summary → Email). Type 'exit' to quit.\n")
    while True:
        doc = input("Paste your document text: ")
        if doc.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = chain.invoke({"text": doc})
        print("\n📧 Final Email Draft:\n", result)
        print("-" * 50)