"""
react_agent_base.py
---------------------
Minimal ReAct-style agent with OpenAI integration, retry/termination logic,
safe calculator tool, real web search tool, and step-by-step logging.

Author: [Your Name]
License: MIT
"""

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import re
import requests


def load_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def load_api_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    if not os.getenv("TAVILY_API_KEY"):
        raise EnvironmentError("TAVILY_API_KEY not found in .env")


def create_messages(prompt: str, user_question: str) -> list:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {user_question}"}
    ]


def call_openai(messages: list, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500, retries=3, delay=2) -> str:
    client = OpenAI()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API call failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("All OpenAI API retry attempts failed.")


def create_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"session_{timestamp}.log"


def log_step(log_path: Path, step: int, content: str):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Step {step}\n{content}\n{'-'*40}\n")


def extract_calculator_expression(reply: str) -> str | None:
    match = re.search(r"Action:\s*Calculator\((.*?)\)", reply)
    return match.group(1) if match else None


def extract_search_query(reply: str) -> str | None:
    match = re.search(r'Action:\s*Search\("(.*?)"\)', reply)
    return match.group(1) if match else None


def safe_eval(expr: str) -> str:
    try:
        allowed_names = {"__builtins__": None}
        result = eval(expr, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def run_search(query: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}"}
        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": 1},
            headers=headers,
            timeout=10
        )
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"][0]["content"]
        else:
            return "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def run_react_loop(prompt: str, user_question: str, max_steps: int = 5):
    messages = create_messages(prompt, user_question)
    log_path = create_log_file()
    print(f"🗂️ Logging steps to: {log_path}\n")

    for step in range(max_steps):
        print(f"\n🔁 Step {step + 1}:")
        reply = call_openai(messages)
        print("🧠 Agent Reply:\n", reply)

        log_step(log_path, step + 1, reply)
        messages.append({"role": "assistant", "content": reply})

        # Handle Calculator
        expr = extract_calculator_expression(reply)
        if expr:
            observation = safe_eval(expr)
            print(f"🔢 Calculator Observation: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # Handle Web Search
        search_query = extract_search_query(reply)
        if search_query:
            observation = run_search(search_query)
            print(f"🌐 Search Observation: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        if "Final Answer:" in reply:
            print("\n✅ Final Answer Detected. Stopping loop.")
            break
    else:
        print("⚠️ Reached max reasoning steps without finding a 'Final Answer:'")


def main():
    print("\U0001F916 ReAct Agent with Calculator + Web Search (A1.1.2.2)")
    load_api_key()

    prompt = load_prompt("prompts/react_base_prompt.txt")
    user_question = input("\U0001F50D Enter your question: ")

    run_react_loop(prompt, user_question)


if __name__ == "__main__":
    main()
