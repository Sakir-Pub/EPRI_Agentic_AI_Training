"""
Hour 4 - Quarter 4: Multi-Agent System Orchestration, Monitoring & Governance (LangChain + MCP)
=================================================================================================

Learning Objectives:
- Enable lifecycle logging and performance tracking in a multi-agent system
- Expand MCP schema to include audit metadata (e.g., trace_id, timestamps)
- Discuss production deployment paths (LangServe, LangGraph preview)
- Simulate system governance via centralized log aggregation

Duration: 15 minutes
"""

import os
import time
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# ============================
# MCP Message Definition (Extended)
# ============================

@dataclass
class MCPMessage:
    sender: str
    recipient: str
    intent: str
    task_type: str
    content: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

# ============================
# Central Audit Log
# ============================

class SystemAuditLog:
    def __init__(self):
        self.entries = []

    def log(self, msg: MCPMessage, result: str, elapsed: float):
        self.entries.append({
            "trace_id": msg.trace_id,
            "sender": msg.sender,
            "recipient": msg.recipient,
            "task": msg.content,
            "output": result,
            "duration_sec": round(elapsed, 2),
            "timestamp": msg.timestamp
        })

    def print_log(self):
        print("\n📊 System Audit Log:")
        for entry in self.entries:
            print(f"- [{entry['trace_id'][:8]}] {entry['recipient']} completed task from {entry['sender']} in {entry['duration_sec']}s")
            print(f"  Task: {entry['task']}\n  → Output: {entry['output']}\n")

# ============================
# Communication + Monitoring
# ============================

class CommunicationHub:
    def __init__(self):
        self.inboxes = {}
        self.responses = {}
        self.audit_log = SystemAuditLog()

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
# Agent Wrapper with Auditing
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
            start = time.time()
            result = self.agent.invoke(msg.content)
            output = result.get('output') if isinstance(result, dict) else result
            elapsed = time.time() - start
            print(f"   🤖 Output: {output} ({round(elapsed, 2)}s)")
            self.hub.store_response(self.name, output)
            self.hub.audit_log.log(msg, output, elapsed)
        self.hub.inboxes[self.name] = []

# ============================
# Orchestrator with Metadata
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

        self.hub.audit_log.print_log()

# ============================
# Demo Setup
# ============================

def dummy_search(q): return f"Trends report on: {q}"
def dummy_calc(e): return str(eval(e))
def dummy_synth(input): return f"Strategic plan based on: {input}"

def run_hour4_q4_workshop():
    print("\n🚀 HOUR 4 - QUARTER 4: SYSTEM ORCHESTRATION, LOGGING + MCP")
    print("=" * 70)

    hub = CommunicationHub()

    agents = {
        "Researcher": MCPAgent("Researcher", "MarketAnalyst", "analyze markets", "Support strategic decisions", [Tool(name="WebSearch", func=dummy_search, description="Market research")], hub),
        "Finance": MCPAgent("Finance", "FinancialExpert", "analyze financials", "Model profitability", [Tool(name="Calculator", func=dummy_calc, description="Math calc")], hub),
        "Strategy": MCPAgent("Strategy", "StrategyConsultant", "synthesize findings", "Recommend business actions", [Tool(name="Synthesizer", func=dummy_synth, description="Strategic synthesis")], hub)
    }

    orchestrator = MCPOrchestrator(agents, hub)

    workflow_tasks = {
        "research": "What are the trends in AI deployment for logistics?",
        "finance": "Given $12M revenue with 17% margin, compute the profit.",
        "strategy": "Use insights to propose a logistics AI expansion roadmap."
    }

    orchestrator.run_workflow(workflow_tasks)
    print("\n✅ QUARTER 4 COMPLETE: Monitoring, auditing, and governance simulated!")

if __name__ == "__main__":
    run_hour4_q4_workshop()
