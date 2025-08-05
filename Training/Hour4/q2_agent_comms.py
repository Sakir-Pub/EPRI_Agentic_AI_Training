"""
Hour 4 - Quarter 2: Agent Communication & Coordination (LangChain + MCP)
===========================================================================

Learning Objectives:
- Enable structured agent-to-agent communication using MCP
- Implement a message protocol with sender, receiver, and task
- Design a simple communication hub to relay messages
- Demonstrate delegation and role-aligned collaboration

Duration: 15 minutes
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

# Load env vars
load_dotenv()

# ============================
# MCP MESSAGE STRUCTURE
# ============================

@dataclass
class MCPMessage:
    sender: str
    recipient: str
    intent: str
    content: str
    context: Dict[str, Any] = None

# ============================
# COMMUNICATION HUB
# ============================

class CommunicationHub:
    def __init__(self):
        self.inboxes = {}

    def register(self, agent_name: str):
        self.inboxes[agent_name] = []

    def send(self, msg: MCPMessage):
        if msg.recipient in self.inboxes:
            self.inboxes[msg.recipient].append(msg)

    def receive(self, agent_name: str) -> List[MCPMessage]:
        return self.inboxes.get(agent_name, [])

    def clear_inbox(self, agent_name: str):
        self.inboxes[agent_name] = []

# ============================
# AGENT WRAPPER WITH MCP
# ============================

class CommunicatingAgent:
    def __init__(self, name: str, role: str, intent: str, goal: str, tools: List[Tool], hub: CommunicationHub):
        self.name = name
        self.hub = hub
        self.hub.register(self.name)

        system_prompt = f"""
[MCP]
role: {role}
intent: {intent}
goal: {goal}
inputs: Messages from other agents

You are an AI agent in a multi-agent system. Your role is to respond only to messages aligned with your intent."
"""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = initialize_agent(
            tools=tools,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            system_message=SystemMessage(content=system_prompt)
        )

    def receive_and_process(self):
        inbox = self.hub.receive(self.name)
        for msg in inbox:
            print(f"\n📩 {self.name} received message from {msg.sender} (intent: {msg.intent})")
            print(f"   Content: {msg.content}")
            response = self.agent.invoke(msg.content)
            print(f"   🤖 {self.name} response:\n   {response.get('output') if isinstance(response, dict) else response}")
        self.hub.clear_inbox(self.name)

# ============================
# DEMO SETUP
# ============================

def dummy_search(query: str) -> str:
    return f"Pretend search result for '{query}'"

def simple_calc(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "Calculation error"

def run_hour4_q2_workshop():
    print("\n🚀 HOUR 4 - QUARTER 2: AGENT COMMUNICATION + MCP")
    print("=" * 70)

    hub = CommunicationHub()

    agents = {
        "Researcher": CommunicatingAgent(
            name="Researcher",
            role="MarketAnalyst",
            intent="respond to research queries",
            goal="Provide insights via web search",
            tools=[Tool(name="WebSearch", func=dummy_search, description="Search the web")],
            hub=hub
        ),
        "Finance": CommunicatingAgent(
            name="Finance",
            role="FinancialExpert",
            intent="answer financial questions",
            goal="Do financial modeling",
            tools=[Tool(name="Calculator", func=simple_calc, description="Basic math")],
            hub=hub
        )
    }

    message1 = MCPMessage(
        sender="Coordinator",
        recipient="Researcher",
        intent="delegate",
        content="What are the key trends in the global semiconductor market?"
    )

    message2 = MCPMessage(
        sender="Coordinator",
        recipient="Finance",
        intent="delegate",
        content="If revenue is $4.2M and profit margin is 18%, what's the profit?"
    )

    hub.send(message1)
    hub.send(message2)

    for agent in agents.values():
        agent.receive_and_process()

    print("\n✅ QUARTER 2 COMPLETE: Agents communicated using MCP messages!")

if __name__ == "__main__":
    run_hour4_q2_workshop()
