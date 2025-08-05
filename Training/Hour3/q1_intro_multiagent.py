"""
Hour 3 - Quarter 1: Introduction to Multi-Agent Systems
=======================================================

Learning Objectives:
- Understand multi-agent system architecture and benefits
- Create specialized agents with distinct roles and expertise
- Learn agent communication and coordination fundamentals
- Build foundation for team-based AI automation

Duration: 15 minutes (after break)
Technical Skills: Agent specialization, inter-agent communication, team architecture
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any

# =============================================================================
# SPECIALIZED AGENT BASE CLASS
# =============================================================================

class SpecializedAgent:
    """
    Base class for specialized agents in multi-agent systems
    Each agent has specific expertise, personality, and capabilities
    """
    
    def __init__(self, agent_id: str, role: str, expertise: str, personality: str):
        """Initialize specialized agent with unique characteristics"""
        load_dotenv()
        self.client = OpenAI()
        self.agent_id = agent_id
        self.role = role
        self.expertise = expertise
        self.personality = personality
        self.conversation_history = []
        self.agent_interactions = []
        
        # Agent-specific system prompt
        self.system_prompt = f"""You are {self.agent_id}, a specialized AI agent in a multi-agent system.

AGENT PROFILE:
- Role: {self.role}
- Expertise: {self.expertise}
- Personality: {self.personality}

MULTI-AGENT INTERACTION RULES:
1. Stay in character and maintain your specialized expertise
2. Collaborate professionally with other agents in the team
3. Share information clearly and request help when needed
4. Build on other agents' contributions to create comprehensive solutions
5. Always identify yourself when communicating with other agents

COMMUNICATION FORMAT:
When working with other agents, always start with:
"[{self.agent_id}]: [Your response]"

COLLABORATION PRINCIPLES:
- Leverage your expertise while respecting others' specializations
- Ask clarifying questions when you need more information
- Provide detailed explanations within your area of expertise
- Suggest involving other agents when their expertise is needed

Your goal is to contribute your specialized knowledge to achieve the team's objectives while maintaining professional collaboration with other agents.
"""
    
    def process_task(self, task: str, context: Dict = None) -> Dict:
        """
        Process a task using the agent's specialized capabilities
        """
        print(f"\n🤖 [{self.agent_id}] processing task...")
        print(f"🎯 Role: {self.role}")
        print(f"🧠 Expertise: {self.expertise}")
        
        # Build message context
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        # Add context from other agents if available
        if context and "agent_inputs" in context:
            context_info = "Previous agent contributions:\n"
            for agent_id, contribution in context["agent_inputs"].items():
                context_info += f"- {agent_id}: {contribution}\n"
            messages.append({"role": "user", "content": context_info})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,  # Moderate creativity for personality
                max_tokens=500
            )
            
            agent_response = response.choices[0].message.content
            print(f"💭 [{self.agent_id}]: {agent_response}")
            
            # Record interaction
            self.conversation_history.append({
                "task": task,
                "response": agent_response,
                "timestamp": datetime.now().isoformat(),
                "context": context
            })
            
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "response": agent_response,
                "expertise_applied": self.expertise,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error in {self.agent_id}: {e}"
            print(f"❌ {error_msg}")
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "response": error_msg,
                "success": False
            }
    
    def collaborate_with(self, other_agent_response: Dict, task: str) -> Dict:
        """
        Collaborate with another agent's output
        """
        collaboration_prompt = f"""Another team member has provided their input:

Agent: {other_agent_response['agent_id']} ({other_agent_response['role']})
Response: {other_agent_response['response']}

Now provide your specialized perspective on the task: {task}

Build on their contribution while adding your unique expertise."""
        
        return self.process_task(collaboration_prompt)

# =============================================================================
# SPECIALIZED AGENT IMPLEMENTATIONS
# =============================================================================

class ResearchAnalyst(SpecializedAgent):
    """Specialized agent for research and data analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id="ResearchAnalyst",
            role="Senior Research Analyst",
            expertise="Market research, data analysis, competitive intelligence, trend identification",
            personality="Analytical, detail-oriented, evidence-based, thorough in research methodology"
        )

class FinancialExpert(SpecializedAgent):
    """Specialized agent for financial analysis and planning"""
    
    def __init__(self):
        super().__init__(
            agent_id="FinancialExpert", 
            role="Senior Financial Analyst",
            expertise="Financial modeling, ROI analysis, budget planning, cost-benefit analysis, risk assessment",
            personality="Precise, numbers-focused, risk-aware, strategic in financial planning"
        )

class StrategyConsultant(SpecializedAgent):
    """Specialized agent for strategic planning and recommendations"""
    
    def __init__(self):
        super().__init__(
            agent_id="StrategyConsultant",
            role="Senior Strategy Consultant", 
            expertise="Strategic planning, business development, market positioning, competitive strategy",
            personality="Big-picture thinker, strategic, decisive, focused on long-term value creation"
        )

class ProjectManager(SpecializedAgent):
    """Specialized agent for project management and coordination"""
    
    def __init__(self):
        super().__init__(
            agent_id="ProjectManager",
            role="Senior Project Manager",
            expertise="Project planning, resource allocation, timeline management, stakeholder coordination",
            personality="Organized, communication-focused, deadline-driven, collaborative team leader"
        )

# =============================================================================
# MULTI-AGENT TEAM COORDINATOR
# =============================================================================

class MultiAgentTeam:
    """
    Coordinator for multi-agent teams
    Manages agent interactions, task distribution, and result synthesis
    """
    
    def __init__(self, team_name: str):
        """Initialize multi-agent team"""
        self.team_name = team_name
        self.agents = {}
        self.team_history = []
        self.collaboration_patterns = {
            "sequential": "Agents work one after another, building on previous outputs",
            "parallel": "Agents work simultaneously on different aspects",
            "collaborative": "Agents work together with cross-communication",
            "hierarchical": "Agents work in structured reporting relationships"
        }
    
    def add_agent(self, agent: SpecializedAgent):
        """Add a specialized agent to the team"""
        self.agents[agent.agent_id] = agent
        print(f"✅ Added {agent.agent_id} ({agent.role}) to {self.team_name}")
    
    def get_team_composition(self) -> Dict:
        """Get overview of team composition and capabilities"""
        composition = {
            "team_name": self.team_name,
            "team_size": len(self.agents),
            "agents": []
        }
        
        for agent in self.agents.values():
            composition["agents"].append({
                "id": agent.agent_id,
                "role": agent.role,
                "expertise": agent.expertise,
                "personality": agent.personality
            })
        
        return composition
    
    def sequential_collaboration(self, task: str, agent_sequence: List[str] = None) -> Dict:
        """
        Execute task using sequential collaboration pattern
        Each agent builds on the previous agent's output
        """
        print(f"\n🔗 {self.team_name}: Sequential Collaboration")
        print(f"📋 Task: {task}")
        print("🔄 Agents will work sequentially, building on each other's contributions\n")
        
        # Use provided sequence or default order
        if not agent_sequence:
            agent_sequence = list(self.agents.keys())
        
        results = []
        context = {"agent_inputs": {}}
        
        for i, agent_id in enumerate(agent_sequence):
            if agent_id not in self.agents:
                print(f"⚠️ Agent {agent_id} not found in team")
                continue
            
            print(f"👥 Step {i+1}: {agent_id} working...")
            
            # First agent gets original task, others get collaborative task
            if i == 0:
                result = self.agents[agent_id].process_task(task, context)
            else:
                # Build collaborative context from previous agents
                collaborative_task = f"Building on the team's previous work, provide your specialized input on: {task}"
                result = self.agents[agent_id].process_task(collaborative_task, context)
            
            results.append(result)
            
            # Add this agent's contribution to context for next agents
            if result["success"]:
                context["agent_inputs"][agent_id] = result["response"]
            
            print("-" * 50)
        
        # Synthesize final team result
        team_result = self._synthesize_team_output(task, results)
        
        # Record team interaction
        self.team_history.append({
            "task": task,
            "collaboration_pattern": "sequential",
            "agents_involved": agent_sequence,
            "results": results,
            "team_synthesis": team_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return team_result
    
    def parallel_collaboration(self, task: str, agent_list: List[str] = None) -> Dict:
        """
        Execute task using parallel collaboration pattern
        All agents work simultaneously on different aspects
        """
        print(f"\n🔀 {self.team_name}: Parallel Collaboration")
        print(f"📋 Task: {task}")
        print("⚡ Agents will work simultaneously on different aspects\n")
        
        # Use provided list or all agents
        if not agent_list:
            agent_list = list(self.agents.keys())
        
        results = []
        
        # All agents work on the task simultaneously
        for agent_id in agent_list:
            if agent_id not in self.agents:
                print(f"⚠️ Agent {agent_id} not found in team")
                continue
            
            print(f"👥 {agent_id} working in parallel...")
            
            # Each agent gets the original task with their expertise focus
            specialized_task = f"From your {self.agents[agent_id].expertise} perspective, analyze: {task}"
            result = self.agents[agent_id].process_task(specialized_task)
            results.append(result)
            
            print("-" * 30)
        
        # Synthesize parallel contributions
        team_result = self._synthesize_team_output(task, results)
        
        # Record team interaction
        self.team_history.append({
            "task": task,
            "collaboration_pattern": "parallel",
            "agents_involved": agent_list,
            "results": results,
            "team_synthesis": team_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return team_result
    
    def _synthesize_team_output(self, original_task: str, agent_results: List[Dict]) -> Dict:
        """
        Synthesize individual agent outputs into cohesive team result
        """
        print("\n🔄 Synthesizing team contributions...")
        
        successful_results = [r for r in agent_results if r["success"]]
        
        if not successful_results:
            return {
                "team_name": self.team_name,
                "task": original_task,
                "success": False,
                "team_synthesis": "No successful agent contributions to synthesize",
                "agent_contributions": len(agent_results)
            }
        
        # Create synthesis of all contributions
        synthesis = f"Team Analysis: {self.team_name}\n"
        synthesis += f"Task: {original_task}\n\n"
        synthesis += "Integrated Team Perspective:\n"
        
        for result in successful_results:
            synthesis += f"\n{result['role']} ({result['agent_id']}):\n"
            synthesis += f"{result['response']}\n"
        
        synthesis += f"\nTeam Conclusion: Combined expertise from {len(successful_results)} specialists provides comprehensive analysis covering multiple business dimensions."
        
        return {
            "team_name": self.team_name,
            "task": original_task,
            "success": True,
            "team_synthesis": synthesis,
            "agent_contributions": len(successful_results),
            "agents_involved": [r["agent_id"] for r in successful_results]
        }

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_multi_agent_concepts():
    """Explain multi-agent system concepts and benefits"""
    print("🤖 UNDERSTANDING MULTI-AGENT SYSTEMS")
    print("=" * 60)
    
    concepts = [
        {
            "concept": "Specialization",
            "description": "Each agent has specific expertise and role",
            "benefit": "Deep domain knowledge and focused capabilities",
            "example": "Financial Expert + Research Analyst + Strategy Consultant"
        },
        {
            "concept": "Collaboration", 
            "description": "Agents work together and share information",
            "benefit": "Combined expertise exceeds individual capabilities",
            "example": "Research feeds into financial analysis feeds into strategy"
        },
        {
            "concept": "Coordination",
            "description": "Systematic management of agent interactions",
            "benefit": "Organized workflow and comprehensive solutions",
            "example": "Project Manager coordinates team and synthesizes results"
        },
        {
            "concept": "Scalability",
            "description": "Easy to add new agents with different expertise",
            "benefit": "Adaptable to complex and changing business needs",
            "example": "Add Legal Expert, Marketing Specialist, or Technical Architect"
        }
    ]
    
    for concept in concepts:
        print(f"\n🔹 {concept['concept']}")
        print(f"   Description: {concept['description']}")
        print(f"   Benefit: {concept['benefit']}")
        print(f"   Example: {concept['example']}")
    
    print("\n🎯 Multi-agent systems enable enterprise-scale AI automation!")

def demonstrate_collaboration_patterns():
    """Show different patterns of agent collaboration"""
    print("\n🔗 COLLABORATION PATTERNS")
    print("=" * 60)
    
    patterns = [
        {
            "pattern": "Sequential Collaboration",
            "description": "Agents work one after another, building on previous outputs",
            "best_for": "Complex analysis requiring multiple perspectives in order",
            "example": "Research → Financial Analysis → Strategic Recommendations"
        },
        {
            "pattern": "Parallel Collaboration",
            "description": "Agents work simultaneously on different aspects",
            "best_for": "Comprehensive analysis requiring multiple viewpoints",
            "example": "All agents analyze acquisition target from their expertise"
        },
        {
            "pattern": "Collaborative Discussion",
            "description": "Agents engage in back-and-forth discussion",
            "best_for": "Complex decisions requiring consensus and refinement",
            "example": "Strategy debate with multiple rounds of input and feedback"
        },
        {
            "pattern": "Hierarchical Coordination",
            "description": "Manager agent coordinates and delegates to specialists",
            "best_for": "Large projects requiring structured management",
            "example": "Project Manager assigns tasks and integrates deliverables"
        }
    ]
    
    for pattern in patterns:
        print(f"\n🔄 {pattern['pattern']}")
        print(f"   How: {pattern['description']}")
        print(f"   Best For: {pattern['best_for']}")
        print(f"   Example: {pattern['example']}")
    
    print("\n🎯 Different patterns optimize for different business scenarios!")

# =============================================================================
# TESTING MULTI-AGENT CAPABILITIES
# =============================================================================

def test_multi_agent_system():
    """Test multi-agent system with business scenarios"""
    print("\n🧪 TESTING MULTI-AGENT TEAM CAPABILITIES")
    print("=" * 60)
    
    # Create specialized team
    print("🏗️ Building Specialized Business Intelligence Team...")
    
    team = MultiAgentTeam("Business Intelligence Team")
    
    # Add specialized agents
    team.add_agent(ResearchAnalyst())
    team.add_agent(FinancialExpert())
    team.add_agent(StrategyConsultant())
    team.add_agent(ProjectManager())
    
    print(f"\n👥 Team Composition:")
    composition = team.get_team_composition()
    for agent in composition["agents"]:
        print(f"   • {agent['id']}: {agent['role']}")
        print(f"     Expertise: {agent['expertise']}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Market Entry Analysis",
            "task": "Should we enter the European software market? Analyze market conditions, financial requirements, competitive landscape, and provide strategic recommendations.",
            "collaboration": "sequential",
            "expected_flow": "Research → Financial → Strategy → Project Planning"
        },
        {
            "name": "Acquisition Evaluation",
            "task": "Evaluate the potential acquisition of a $50M AI startup. Consider market position, financial implications, strategic fit, and implementation challenges.",
            "collaboration": "parallel",
            "expected_flow": "All agents analyze simultaneously from their expertise"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Multi-Agent Test {i}: {scenario['name']}")
        print(f"🔄 Collaboration Pattern: {scenario['collaboration']}")
        print(f"📈 Expected Flow: {scenario['expected_flow']}")
        print(f"❓ Scenario: {scenario['task']}")
        
        # Execute based on collaboration pattern
        if scenario['collaboration'] == 'sequential':
            result = team.sequential_collaboration(scenario['task'])
        else:
            result = team.parallel_collaboration(scenario['task'])
        
        print(f"\n🏆 Team Result Summary:")
        print(f"   Success: {result['success']}")
        print(f"   Agents Involved: {len(result['agents_involved'])}")
        print(f"   Team Analysis: {result['team_synthesis'][:200]}...")
        
        print("\n" + "=" * 80)
        
        if i < len(test_scenarios):
            input("Press Enter to continue to next multi-agent test...")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def multi_agent_workshop():
    """Interactive workshop with multi-agent teams"""
    print("\n🎯 MULTI-AGENT SYSTEM WORKSHOP")
    print("=" * 60)
    
    # Create team for workshop
    team = MultiAgentTeam("Workshop Team")
    team.add_agent(ResearchAnalyst())
    team.add_agent(FinancialExpert())
    team.add_agent(StrategyConsultant())
    team.add_agent(ProjectManager())
    
    print("Test your multi-agent team with complex business scenarios!")
    print("Multi-Agent Challenges:")
    print("• Strategic planning requiring multiple expertise areas")
    print("• Investment decisions needing research + financial + strategic analysis")
    print("• Market expansion scenarios requiring comprehensive evaluation")
    print("• Complex business problems benefiting from team collaboration")
    print("\nCollaboration options: 'sequential' or 'parallel'")
    print("Type 'exit' to finish this quarter.")
    
    while True:
        print(f"\n👥 Current Team: {len(team.agents)} specialized agents")
        user_task = input("\n💬 Your multi-agent scenario: ")
        
        if user_task.lower() in ['exit', 'quit', 'done']:
            print("🎉 Excellent! You've built and tested multi-agent systems!")
            break
        
        if not user_task.strip():
            print("Please enter a business scenario for the team to analyze.")
            continue
        
        # Ask for collaboration pattern
        pattern = input("Choose collaboration pattern (sequential/parallel): ").lower()
        if pattern not in ['sequential', 'parallel']:
            pattern = 'sequential'  # default
        
        print(f"\n🔄 Executing {pattern} collaboration...")
        
        # Execute team collaboration
        if pattern == 'sequential':
            result = team.sequential_collaboration(user_task)
        else:
            result = team.parallel_collaboration(user_task)
        
        print(f"\n🎯 Multi-Agent Team Result:")
        print(f"Success: {result['success']}")
        print(f"Team Analysis: {result['team_synthesis'][:300]}...")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour3_q1_workshop():
    """Main function for Hour 3 Q1 workshop"""
    print("🚀 HOUR 3 - QUARTER 1: INTRODUCTION TO MULTI-AGENT SYSTEMS")
    print("=" * 70)
    print("Welcome to the next frontier: Multi-Agent Intelligence Teams!\n")
    
    # Step 1: Explain multi-agent concepts
    demonstrate_multi_agent_concepts()
    
    # Step 2: Show collaboration patterns
    demonstrate_collaboration_patterns()
    
    # Step 3: Test multi-agent capabilities
    test_multi_agent_system()
    
    # Step 4: Interactive workshop
    multi_agent_workshop()
    
    # Step 5: Quarter completion and Q2 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 1 COMPLETE!")
    print("=" * 60)
    print("Multi-Agent System Achievements:")
    print("✅ Understanding of multi-agent architecture and benefits")
    print("✅ Created specialized agents with distinct roles and expertise")
    print("✅ Implemented agent communication and coordination")
    print("✅ Built foundation for team-based AI automation")
    print("✅ Tested sequential and parallel collaboration patterns")
    
    print("\n🏆 Your Multi-Agent Capabilities:")
    print("   → Specialized agent teams with distinct expertise")
    print("   → Sequential collaboration for building complex analysis")
    print("   → Parallel collaboration for comprehensive evaluation")
    print("   → Team coordination and result synthesis")
    print("   → Scalable architecture for adding new agent specializations")
    
    print("\n📈 Evolution Summary:")
    print("   Hour 1-2: Individual agent mastery")
    print("   Hour 3 Q1: Multi-agent team foundations")
    print("   Hour 3 Q2: Advanced communication & coordination (coming next)")
    
    print("\n🚀 Coming Up in Q2: Agent Communication & Coordination")
    print("   → Advanced inter-agent communication protocols")
    print("   → Dynamic task delegation and workload balancing")
    print("   → Real-time agent collaboration and negotiation")
    print("   → Intelligent agent selection and team composition")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 3 Q2: Advanced Agent Communication!")

if __name__ == "__main__":
    # Run the complete Hour 3 Q1 workshop
    run_hour3_q1_workshop()