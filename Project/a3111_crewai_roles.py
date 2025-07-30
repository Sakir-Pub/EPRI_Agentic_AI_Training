# a3_crewai_roles.py

from crewai import Agent

# Define the Planner agent
planner = Agent(
    role="Planner",
    goal="Read a document and create a plan to summarize and email it.",
    backstory="You're an experienced AI workflow planner who breaks down document workflows.",
    verbose=True
)

# Define the Worker agent
worker = Agent(
    role="Worker",
    goal="Summarize a document and draft an email from it.",
    backstory="You're a skilled language model that writes summaries and emails from documents.",
    verbose=True
)

# Define the Reviewer agent
reviewer = Agent(
    role="Reviewer",
    goal="Evaluate the quality of the summary and email, and provide improvements.",
    backstory="You're a senior AI reviewer who ensures output quality and tone.",
    verbose=True
)

# Just print agent summaries to verify setup
if __name__ == "__main__":
    print("✅ Agent roles created:\n")
    for agent in [planner, worker, reviewer]:
        print(f"{agent.role}: {agent.goal}")
