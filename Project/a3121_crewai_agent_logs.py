# a3_crewai_multistep_logging.py

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()
console = Console()

# Define agents
planner = Agent(
    role="Planner",
    goal="Devise a plan to summarize and email a document",
    backstory="An expert AI strategist for knowledge workflows.",
    verbose=True
)

worker = Agent(
    role="Worker",
    goal="Summarize the document and write a professional email",
    backstory="A writing assistant specialized in technical summaries and formal communication.",
    verbose=True
)

reviewer = Agent(
    role="Reviewer",
    goal="Review and enhance the worker's output for clarity and professionalism",
    backstory="A meticulous editor focused on language quality.",
    verbose=True
)

document = """
Our team recently completed the development phase for the new analytics dashboard.
Key achievements include reducing load time by 40%, improving user satisfaction scores,
and integrating new real-time data features. Next steps include launching in beta and
gathering early user feedback. Risks include potential delays in backend services.
"""

# Define tasks
task1 = Task(
    description=f"""Given the document:

\"\"\"{document}\"\"\"

Devise a step-by-step plan to summarize it and create a professional email.""",
    expected_output="Step-by-step summary-email plan",
    agent=planner,
)

task2 = Task(
    description="Follow the planner's strategy to generate a summary and formal email.",
    expected_output="Summary paragraph + professional email draft.",
    agent=worker,
)

task3 = Task(
    description="Review and improve the email and summary written by the worker.",
    expected_output="Improved version of both pieces with refined language and tone.",
    agent=reviewer,
)

# Run one task at a time so we can log after each
crew = Crew(
    agents=[planner, worker, reviewer],
    tasks=[task1, task2, task3],
    verbose=True
)

if __name__ == "__main__":
    outputs = crew._execute_tasks([task1, task2, task3])  # internal method to run sequentially and get outputs

    console.rule("[bold blue]🔷 Planner's Output")
    console.print(Markdown(outputs[0]))

    console.rule("[bold green]✍️ Worker's Output")
    console.print(Markdown(outputs[1]))

    console.rule("[bold yellow]✅ Reviewer's Final Output")
    console.print(Markdown(outputs[2]))
