"""
Hour 4 - Quarter 1: Specialized Agent Creation (LangChain + MCP)
==================================================================

Learning Objectives:
- Create specialized agents using LangChain
- Define static MCP context for each agent
- Enable role-specific reasoning and tools
- Build scalable agent factory for future use

Duration: 15 minutes

Note: This file uses LangChain 0.1.x conventions. Some APIs (like `AgentExecutor`, `run()`, and `ChatOpenAI`) are deprecated as of LangChain 0.2.0+. For future-proofing, consider switching to LangGraph and the langchain-openai/langchain-community packages.
"""

import os
from dotenv import load_dotenv
from typing import Dict
from langchain_openai import ChatOpenAI  # updated import per deprecation warning
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

# Load environment
load_dotenv()

# ============================
# MCP ROLE PROMPT TEMPLATES
# ============================

def generate_mcp_prompt(role: str, intent: str, goal: str) -> str:
    return f"""
[MCP]
role: {role}
intent: {intent}
goal: {goal}
inputs: User query or task request

You are a specialized AI agent operating under the Model Context Protocol (MCP). Your role, intent, and goal are defined above. Act accordingly and only respond based on your assigned responsibilities.
"""

# ============================
# TOOL DEFINITIONS
# ============================

def calculator_tool_func(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return f"Calculation error: {e}"

def dummy_search_tool_func(query: str) -> str:
    return f"Pretend search results for: '{query}'"

calculator_tool = Tool(
    name="Calculator",
    func=calculator_tool_func,
    description="Useful for performing basic math operations or evaluating numeric expressions."
)

search_tool = Tool(
    name="WebSearch",
    func=dummy_search_tool_func,
    description="Useful for searching current information or market data."
)

# ============================
# MCP AGENT CREATION
# ============================

def create_mcp_agent(role: str, intent: str, goal: str, tools: list) -> Dict:
    mcp_prompt = generate_mcp_prompt(role, intent, goal)
    system_message = SystemMessage(content=mcp_prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        system_message=system_message
    )

    return {
        "agent": agent,
        "role": role,
        "intent": intent,
        "goal": goal
    }

# ============================
# AGENT REGISTRY
# ============================

def create_specialized_mcp_agents():
    """Return dictionary of preconfigured MCP-based LangChain agents"""
    agents = {
        "MarketAnalyst": create_mcp_agent(
            role="MarketAnalyst",
            intent="research trends and competitors",
            goal="Provide market intelligence for strategic decisions",
            tools=[search_tool]
        ),
        "FinancialExpert": create_mcp_agent(
            role="FinancialExpert",
            intent="perform financial modeling and ROI analysis",
            goal="Support investment and budgeting decisions",
            tools=[calculator_tool]
        ),
        "StrategyConsultant": create_mcp_agent(
            role="StrategyConsultant",
            intent="synthesize insights and make strategic recommendations",
            goal="Deliver comprehensive business strategies",
            tools=[search_tool, calculator_tool]
        )
    }
    return agents

# ============================
# DEMO + TEST CASES
# ============================

def run_agent_tests():
    agents = create_specialized_mcp_agents()
    print("\n🧪 Running predefined MCP agent tests...")

    test_tasks = {
        "MarketAnalyst": [
            "What are the current trends in the AI hardware market?",
            "List top 3 competitors in the cloud AI space."
        ],
        "FinancialExpert": [
            "What is the ROI of a $2M investment returning $2.6M over 2 years?",
            "Evaluate the profitability of a 15% margin on $500,000 revenue."
        ],
        "StrategyConsultant": [
            "How should we expand into the Asia-Pacific market?",
            "Combine market insights and financials to propose a growth strategy."
        ]
    }

    for agent_name, tasks in test_tasks.items():
        config = agents[agent_name]
        agent = config["agent"]

        print(f"\n🤖 Testing Agent: {agent_name}")
        print("-" * 60)
        for task in tasks:
            print(f"\n💬 Task: {task}")
            print("🤔 Thinking...")
            response = agent.invoke(task)  # updated from run() to invoke()
            print(f"🎯 Response:\n{response}\n")

# ============================
# WORKSHOP DEMO FUNCTION
# ============================

def run_hour4_q1_workshop():
    print("\n🚀 HOUR 4 - QUARTER 1: SPECIALIZED AGENT CREATION (MCP + LangChain)")
    print("=" * 80)

    agents = create_specialized_mcp_agents()

    print("\n📌 Available Agents:")
    for name in agents:
        print(f" - {name}")

    while True:
        agent_name = input("\n🔎 Choose an agent (or type 'exit'): ").strip()
        if agent_name.lower() in ['exit', 'quit']: break
        if agent_name not in agents:
            print("❌ Agent not found. Please choose a valid agent.")
            continue

        task = input("💬 Enter a task for this agent: ").strip()
        if not task:
            print("❗ Please provide a valid task.")
            continue

        print("🤖 Agent is processing...")
        response = agents[agent_name]["agent"].invoke(task)  # updated from run() to invoke()
        print(f"\n🎯 Response:\n{response}\n")

    print("\n✅ QUARTER 1 COMPLETE: Specialized Agents Initialized with MCP!")

# ============================
# MAIN
# ============================

if __name__ == "__main__":
    run_agent_tests()
    run_hour4_q1_workshop()