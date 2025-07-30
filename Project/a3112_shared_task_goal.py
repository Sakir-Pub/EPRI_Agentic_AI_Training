# a3_crewai_multistep.py

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()  # Load OPENAI_API_KEY from .env file

# Step 1: Define Agents
planner = Agent(
    role="Planner",
    goal="Devise a step-by-step approach to summarize a document and write an email.",
    backstory="An expert AI strategist known for clear, actionable plans.",
    verbose=True
)

worker = Agent(
    role="Worker",
    goal="Generate a summary and a formal email based on planner instructions.",
    backstory="A language model agent specialized in professional writing and summarization.",
    verbose=True
)

reviewer = Agent(
    role="Reviewer",
    goal="Review and enhance the email and summary written by the worker.",
    backstory="A senior editor ensuring clarity, professionalism, and correctness.",
    verbose=True
)

# Sample input document (you can make this dynamic later)
document = """
Our team recently completed the development phase for the new analytics dashboard.
Key achievements include reducing load time by 40%, improving user satisfaction scores,
and integrating new real-time data features. Next steps include launching in beta and
gathering early user feedback. Risks include potential delays in backend services.
"""

# Step 2: Define Tasks
task1 = Task(
    description=f"""You are given the following document:

\"\"\"{document}\"\"\"

Devise a clear plan to summarize this document and turn it into a professional email. List the steps the worker should follow.""",
    expected_output="Step-by-step plan in plain English",
    agent=planner
)

task2 = Task(
    description="Using the planner's step-by-step instructions, create a concise summary and write a professional email conveying the key points.",
    expected_output="A short paragraph summary and a formal email draft.",
    agent=worker
)

task3 = Task(
    description="Review the summary and email draft from the worker. Improve clarity, tone, grammar, and professionalism.",
    expected_output="Final polished version of the summary and email.",
    agent=reviewer
)

# Step 3: Create and Run the Crew
crew = Crew(
    agents=[planner, worker, reviewer],
    tasks=[task1, task2, task3],
    verbose=True
)

if __name__ == "__main__":
    final_output = crew.kickoff()
    print("\n🧠 Final Output After Review:\n")
    print(final_output)
