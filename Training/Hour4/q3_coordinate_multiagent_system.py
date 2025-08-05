"""
Hour 4 - Quarter 3: Coordinated Multi-Agent System (LangChain + MCP)
======================================================================

Learning Objectives:
- Coordinate specialized agents using a central orchestrator
- Use MCP metadata to guide task routing and result collection
- Aggregate agent responses into a synthesized output
- Maintain modular and extensible system architecture

Duration: 15 minutes
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# ============================
# MCP Message Definition
# ============================

@dataclass
class MCPMessage:
    sender: str
    recipient: str
    intent: str
    task_type: str
    content: str
    context: Dict[str, Any] = None

# ============================
# Communication + Orchestration
# ============================

class CommunicationHub:
    def __init__(self):
        self.inboxes = {}
        self.responses = {}

    def register(self, agent_name: str):
        self.inboxes[agent_name] = []
        self.responses[agent_name] = []

    def send(self, msg: MCPMessage):
        if msg.recipient in self.inboxes:
            self.inboxes[msg.recipient].append(msg)

    def receive(self, agent_name: str) -> List[MCPMessage]:
        return self.inboxes.get(agent_name, [])

    def store_response(self, agent_name: str, response: str):
        self.responses[agent_name].append(response)

    def get_all_responses(self) -> Dict[str, List[str]]:
        return self.responses

    def clear(self):
        for k in self.inboxes:
            self.inboxes[k] = []
            self.responses[k] = []

# ============================
# Agent Wrapper
# ============================

class MCPAgent:
    def __init__(self, name, role, intent, goal, tools, hub: CommunicationHub):
        self.name = name
        self.hub = hub
        self.hub.register(self.name)

        system_prompt = f"""
[MCP]
role: {role}
intent: {intent}
goal: {goal}
inputs: Task requests routed by an orchestrator
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

    def process_messages(self):
        inbox = self.hub.receive(self.name)
        for msg in inbox:
            print(f"\n📩 {self.name} received: {msg.task_type.upper()} from {msg.sender}")
            print(f"   Task: {msg.content}")
            result = self.agent.invoke(msg.content)
            output = result.get('output') if isinstance(result, dict) else result
            print(f"   🤖 Output: {output}")
            self.hub.store_response(self.name, output)
        self.hub.inboxes[self.name] = []

# ============================
# Orchestrator Logic
# ============================

class MCPOrchestrator:
    def __init__(self, agents: Dict[str, MCPAgent], hub: CommunicationHub):
        self.agents = agents
        self.hub = hub

    def route_task(self, task: str, task_type: str):
        routing_map = {
            "research": "Researcher",
            "finance": "Finance",
            "strategy": "Strategy"
        }
        recipient = routing_map.get(task_type)
        if recipient in self.agents:
            msg = MCPMessage(
                sender="Orchestrator",
                recipient=recipient,
                intent="delegate",
                task_type=task_type,
                content=task
            )
            self.hub.send(msg)

    def run_workflow(self, tasks: Dict[str, str]):
        print("\n🧭 Orchestrator dispatching tasks...")
        for task_type, content in tasks.items():
            self.route_task(content, task_type)
        for agent in self.agents.values():
            agent.process_messages()

        print("\n🧾 Aggregated Results:")
        for agent_name, outputs in self.hub.get_all_responses().items():
            print(f"\n🔹 {agent_name}:")
            for out in outputs:
                print(f"   - {out}")

# ============================
# Demo Setup
# ============================

def dummy_search(q): return f"Trends report on: {q}"
def dummy_calc(e): return str(eval(e))
def dummy_synth(input): return f"Strategic plan based on: {input}"

def run_hour4_q3_workshop():
    print("\n🚀 HOUR 4 - QUARTER 3: COORDINATED MULTI-AGENT SYSTEM + MCP")
    print("=" * 70)

    hub = CommunicationHub()

    agents = {
        "Researcher": MCPAgent("Researcher", "MarketAnalyst", "analyze markets", "Support strategic decisions", [Tool(name="WebSearch", func=dummy_search, description="Market research")], hub),
        "Finance": MCPAgent("Finance", "FinancialExpert", "analyze financials", "Model profitability", [Tool(name="Calculator", func=dummy_calc, description="Math calc")], hub),
        "Strategy": MCPAgent("Strategy", "StrategyConsultant", "synthesize findings", "Recommend business actions", [Tool(name="Synthesizer", func=dummy_synth, description="Strategic synthesis")], hub)
    }

    orchestrator = MCPOrchestrator(agents, hub)

    workflow_tasks = {
        "research": "What are the trends in enterprise AI adoption in manufacturing?",
        "finance": "Revenue is $10M and margin is 22%. What's the profit?",
        "strategy": "Combine insights into a single growth strategy"
    }

    orchestrator.run_workflow(workflow_tasks)
    print("\n✅ QUARTER 3 COMPLETE: Multi-agent coordination and aggregation done!")

if __name__ == "__main__":
    run_hour4_q3_workshop()
