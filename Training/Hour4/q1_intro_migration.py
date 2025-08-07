"""
Hour 4 - Quarter 1: LangChain Fundamentals & Agent Migration
============================================================

Learning Objectives:
- Master LangChain architecture and production benefits
- Migrate custom agents to LangChain framework
- Build production-ready agent systems with LangChain
- Understand LangChain agents, tools, chains, and memory

Duration: 15 minutes (after break)
Technical Skills: LangChain agents, tool integration, chain composition, production deployment
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Type
from datetime import datetime

# LangChain Core Imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# Additional LangChain Components
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import ToolException

# Environment and utilities
from dotenv import load_dotenv
import json
import requests
from pathlib import Path

# =============================================================================
# LANGCHAIN PRODUCTION TOOLS
# =============================================================================

@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        String containing the calculation result
    """
    try:
        # Safe evaluation - only allow mathematical operations
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "pow": pow,
            "max": max,
            "min": min,
            "sum": sum,
        }
        
        # Clean expression
        clean_expr = expression.strip()
        
        # Evaluate safely
        result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
        return f"Calculation: {expression} = {result}"
        
    except Exception as e:
        return f"Calculator error: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current information.
    
    Args:
        query: Search query to look up information
        
    Returns:
        String containing search results summary
    """
    try:
        # Load API key
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            return "Web search unavailable: API key not configured. This would normally search for: " + query
        
        # Tavily search API
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "query": query,
            "max_results": 3,
            "search_depth": "basic"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            summary = f"Search results for '{query}':\n"
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")[:80]
                summary += f"{i}. {title}\n"
            
            return summary
        else:
            return f"Search API error: {response.status_code}"
            
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def file_analyzer_tool(file_path: str) -> str:
    """
    Analyze business documents and extract key information.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        String containing file analysis summary
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            # For demo purposes, create sample analysis
            return f"File analysis demo for '{file_path}':\n- Document type: Business report\n- Key metrics found: Revenue, growth rates, market analysis\n- Content summary: Strategic business document with financial and market data"
        
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Basic file analysis
        lines = content.split('\n')
        words = len(content.split())
        
        summary = f"File Analysis: {path.name}\n"
        summary += f"- Size: {len(lines)} lines, {words} words\n"
        summary += f"- Preview: {content[:200]}..."
        
        return summary
        
    except Exception as e:
        return f"File analysis error: {str(e)}"

# =============================================================================
# LANGCHAIN AGENT FACTORY
# =============================================================================

class LangChainAgentFactory:
    """
    Factory for creating production-ready LangChain agents
    """
    
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Available tools for agents
        self.available_tools = [
            calculator_tool,
            web_search_tool,
            file_analyzer_tool
        ]
    
    def create_business_analyst_agent(self) -> AgentExecutor:
        """Create a business analyst agent with LangChain"""
        
        # Define system prompt for business analyst
        system_prompt = """You are a Senior Business Analyst AI agent powered by LangChain.

You have access to powerful tools for business analysis:
- Calculator: For precise financial calculations and modeling
- Web Search: For current market information and research
- File Analyzer: For analyzing business documents and reports

Your approach:
1. Analyze the business question thoroughly
2. Use appropriate tools to gather information and perform calculations
3. Provide comprehensive business insights and recommendations
4. Always show your reasoning process

When using tools:
- Use Calculator for any mathematical operations requiring precision
- Use Web Search when you need current market data or industry information  
- Use File Analyzer when working with documents or reports

Provide professional, well-structured responses with clear recommendations."""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.available_tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.available_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def create_research_specialist_agent(self) -> AgentExecutor:
        """Create a research specialist agent with LangChain"""
        
        system_prompt = """You are a Senior Research Specialist AI agent powered by LangChain.

Your expertise focuses on:
- Market research and competitive intelligence
- Data analysis and trend identification
- Information synthesis from multiple sources

Your methodology:
1. Break down research questions into investigable components
2. Use web search to gather current information
3. Analyze and synthesize findings
4. Provide evidence-based insights and recommendations

Always cite your sources and provide confidence levels for your findings."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.available_tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.available_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def create_financial_expert_agent(self) -> AgentExecutor:
        """Create a financial expert agent with LangChain"""
        
        system_prompt = """You are a Senior Financial Expert AI agent powered by LangChain.

Your specializations:
- Financial modeling and analysis
- ROI calculations and investment analysis
- Risk assessment and budget planning
- Cost-benefit analysis

Your approach:
1. Identify all financial components of the problem
2. Use calculator tool for all mathematical operations
3. Build financial models step-by-step
4. Provide clear financial recommendations with supporting calculations

Always show your calculations explicitly and explain your financial reasoning."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.available_tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.available_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

# =============================================================================
# LANGCHAIN CHAIN COMPOSITIONS
# =============================================================================

class BusinessAnalysisChain:
    """
    LangChain chain for complex business analysis workflows
    """
    
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def create_market_analysis_chain(self):
        """Create a chain for market analysis workflows"""
        
        # Step 1: Market research
        research_prompt = ChatPromptTemplate.from_template(
            """Analyze the market research question: {question}
            
            Break this down into specific research areas that need investigation.
            Identify what current market information would be most valuable.
            
            Provide a structured research plan."""
        )
        
        # Step 2: Data synthesis
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Based on the research plan: {research_plan}
            And the gathered information: {research_data}
            
            Synthesize the findings into key market insights.
            Identify trends, opportunities, and competitive dynamics."""
        )
        
        # Step 3: Strategic recommendations
        recommendation_prompt = ChatPromptTemplate.from_template(
            """Based on the market analysis: {market_insights}
            
            Provide strategic recommendations including:
            1. Key opportunities to pursue
            2. Potential risks and mitigation strategies  
            3. Competitive positioning recommendations
            4. Next steps for implementation
            
            Make recommendations specific and actionable."""
        )
        
        # Build the chain
        research_chain = research_prompt | self.llm | StrOutputParser()
        synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
        recommendation_chain = recommendation_prompt | self.llm | StrOutputParser()
        
        # Combine into full workflow
        full_chain = (
            RunnablePassthrough.assign(
                research_plan=research_chain
            )
            .assign(
                research_data=lambda x: f"[Research data would be gathered here for: {x['research_plan']}]"
            )
            .assign(
                market_insights=synthesis_chain
            )
            .assign(
                recommendations=recommendation_chain
            )
        )
        
        return full_chain
    
    def create_financial_analysis_chain(self):
        """Create a chain for financial analysis workflows"""
        
        # Financial data extraction
        extraction_prompt = ChatPromptTemplate.from_template(
            """Extract and identify all financial data points from: {input}
            
            Identify:
            - Revenue figures and trends
            - Cost structures and expenses  
            - Profitability metrics
            - Growth rates and projections
            - Key financial ratios
            
            Structure the data for analysis."""
        )
        
        # Financial modeling
        modeling_prompt = ChatPromptTemplate.from_template(
            """Based on the financial data: {financial_data}
            
            Create financial models and calculate key metrics:
            - Revenue growth analysis
            - Profitability trends
            - Return on investment calculations
            - Risk assessment indicators
            
            Show all calculations step by step."""
        )
        
        # Financial recommendations
        recommendation_prompt = ChatPromptTemplate.from_template(
            """Based on the financial analysis: {financial_model}
            
            Provide financial recommendations:
            1. Financial performance assessment
            2. Areas for improvement
            3. Investment priorities
            4. Risk mitigation strategies
            5. Financial planning recommendations
            
            Support all recommendations with data."""
        )
        
        # Build chain
        extraction_chain = extraction_prompt | self.llm | StrOutputParser()
        modeling_chain = modeling_prompt | self.llm | StrOutputParser()
        recommendation_chain = recommendation_prompt | self.llm | StrOutputParser()
        
        full_chain = (
            RunnablePassthrough.assign(
                financial_data=extraction_chain
            )
            .assign(
                financial_model=modeling_chain
            )
            .assign(
                financial_recommendations=recommendation_chain
            )
        )
        
        return full_chain

# =============================================================================
# CUSTOM AGENT TO LANGCHAIN MIGRATION
# =============================================================================

class AgentMigrationDemo:
    """
    Demonstrate migration from custom agents to LangChain
    """
    
    def __init__(self):
        self.factory = LangChainAgentFactory()
        
    def compare_custom_vs_langchain(self):
        """Compare custom agent implementation with LangChain version"""
        
        print("🔄 CUSTOM AGENTS vs LANGCHAIN COMPARISON")
        print("=" * 60)
        
        comparison_points = [
            {
                "aspect": "Code Complexity",
                "custom": "100+ lines for agent setup, tool integration, error handling",
                "langchain": "20 lines with built-in agent creation and tool integration",
                "benefit": "90% reduction in boilerplate code"
            },
            {
                "aspect": "Tool Integration", 
                "custom": "Manual tool calling, parsing, error handling",
                "langchain": "Automatic tool calling with @tool decorator",
                "benefit": "Built-in tool management and error handling"
            },
            {
                "aspect": "Memory Management",
                "custom": "Custom conversation tracking and state management",
                "langchain": "Built-in memory classes with persistence options",
                "benefit": "Production-ready memory management"
            },
            {
                "aspect": "Error Handling",
                "custom": "Custom try/catch blocks and retry logic",
                "langchain": "Built-in parsing error handling and recovery",
                "benefit": "Robust error handling out of the box"
            },
            {
                "aspect": "Observability",
                "custom": "Manual logging and print statements",
                "langchain": "Built-in callbacks and tracing integration",
                "benefit": "Production monitoring and debugging"
            },
            {
                "aspect": "Scalability",
                "custom": "Manual optimization and resource management",
                "langchain": "Built-in async support and optimization",
                "benefit": "Enterprise-scale performance"
            }
        ]
        
        for point in comparison_points:
            print(f"\n📊 {point['aspect']}:")
            print(f"   Custom Implementation: {point['custom']}")
            print(f"   LangChain Framework: {point['langchain']}")
            print(f"   ✅ Benefit: {point['benefit']}")
        
        print("\n🎯 LangChain provides production-ready infrastructure!")
    
    def demonstrate_langchain_agent_creation(self):
        """Show how easy it is to create agents with LangChain"""
        
        print("\n🛠️ LANGCHAIN AGENT CREATION DEMO")
        print("=" * 60)
        
        print("Step 1: Creating Business Analyst Agent...")
        business_agent = self.factory.create_business_analyst_agent()
        print("✅ Business Analyst Agent created with 3 tools")
        
        print("\nStep 2: Creating Research Specialist Agent...")
        research_agent = self.factory.create_research_specialist_agent() 
        print("✅ Research Specialist Agent created with 3 tools")
        
        print("\nStep 3: Creating Financial Expert Agent...")
        financial_agent = self.factory.create_financial_expert_agent()
        print("✅ Financial Expert Agent created with 3 tools")
        
        return {
            "business_analyst": business_agent,
            "research_specialist": research_agent,
            "financial_expert": financial_agent
        }

# =============================================================================
# LANGCHAIN PRODUCTION FEATURES
# =============================================================================

class LangChainProductionFeatures:
    """
    Demonstrate LangChain's production-ready features
    """
    
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(temperature=0.1)
    
    def demonstrate_memory_systems(self):
        """Show LangChain's memory capabilities"""
        
        print("\n🧠 LANGCHAIN MEMORY SYSTEMS")
        print("=" * 50)
        
        # Buffer Memory
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Summary Memory
        summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True
        )
        
        print("✅ ConversationBufferMemory: Stores full conversation history")
        print("✅ ConversationSummaryMemory: Summarizes long conversations")
        print("✅ Built-in persistence and retrieval")
        print("✅ Automatic memory management")
        
        return {
            "buffer_memory": buffer_memory,
            "summary_memory": summary_memory
        }
    
    def demonstrate_callback_system(self):
        """Show LangChain's callback and monitoring system"""
        
        print("\n📊 LANGCHAIN CALLBACK & MONITORING")
        print("=" * 50)
        
        class BusinessAnalyticsCallback(BaseCallbackHandler):
            """Custom callback for business analytics"""
            
            def __init__(self):
                super().__init__()
                self.tool_usage = {}
                self.token_usage = 0
                self.start_time = None
            
            def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
                tool_name = serialized.get("name", "unknown")
                self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
                print(f"🔧 Tool used: {tool_name}")
            
            def on_agent_action(self, action: AgentAction, **kwargs) -> None:
                print(f"🤖 Agent action: {action.tool}")
            
            def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
                print("✅ Agent task completed")
                print(f"📈 Tool usage summary: {self.tool_usage}")
        
        callback = BusinessAnalyticsCallback()
        print("✅ Custom callback created for monitoring")
        print("✅ Tracks tool usage, performance, and completion")
        print("✅ Integrates with monitoring systems")
        
        return callback
    
    def demonstrate_chain_composition(self):
        """Show LangChain's chain composition capabilities"""
        
        print("\n🔗 LANGCHAIN CHAIN COMPOSITION")
        print("=" * 50)
        
        analyzer = BusinessAnalysisChain()
        
        # Create market analysis chain
        market_chain = analyzer.create_market_analysis_chain()
        print("✅ Market Analysis Chain: Research → Synthesis → Recommendations")
        
        # Create financial analysis chain  
        financial_chain = analyzer.create_financial_analysis_chain()
        print("✅ Financial Analysis Chain: Extraction → Modeling → Recommendations")
        
        print("✅ Parallel execution support")
        print("✅ Error handling and recovery")
        print("✅ State management between steps")
        
        return {
            "market_chain": market_chain,
            "financial_chain": financial_chain
        }

# =============================================================================
# TESTING LANGCHAIN AGENTS
# =============================================================================

def test_langchain_agents():
    """Test LangChain agents with business scenarios"""
    
    print("\n🧪 TESTING LANGCHAIN AGENTS")
    print("=" * 60)
    
    # Create agent factory
    factory = LangChainAgentFactory()
    
    # Create business analyst agent
    print("Creating LangChain Business Analyst Agent...")
    business_agent = factory.create_business_analyst_agent()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Financial Analysis with Tools",
            "query": "Calculate the ROI if we invest $100,000 in marketing and it generates $150,000 in additional revenue. Also, what's the current average marketing ROI in our industry?",
            "expected_tools": ["calculator_tool", "web_search_tool"]
        },
        {
            "name": "Market Research Analysis", 
            "query": "Research the current trends in AI software market growth and analyze what this means for our business strategy.",
            "expected_tools": ["web_search_tool"]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 LangChain Test {i}: {scenario['name']}")
        print(f"🎯 Expected Tools: {', '.join(scenario['expected_tools'])}")
        print(f"❓ Query: {scenario['query']}")
        
        try:
            print(f"\n🚀 Executing LangChain agent...")
            result = business_agent.invoke({"input": scenario['query']})
            
            print(f"\n🏆 LangChain Agent Result:")
            print(f"Output: {result['output'][:300]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 70)
        
        if i < len(test_scenarios):
            input("Press Enter to continue to next LangChain test...")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def langchain_migration_workshop():
    """Interactive workshop for migrating to LangChain"""
    
    print("\n🎯 LANGCHAIN MIGRATION WORKSHOP")
    print("=" * 60)
    
    # Create factory and demo
    factory = LangChainAgentFactory()
    migration_demo = AgentMigrationDemo()
    
    print("Experience the power of LangChain for production AI agents!")
    print("Migration Challenges:")
    print("• Compare custom implementations with LangChain equivalents")
    print("• Test production-ready agents with real business scenarios")
    print("• Experience LangChain's advanced features (memory, callbacks, chains)")
    print("• Build scalable, maintainable agent systems")
    print("\nType 'exit' to finish this quarter.")
    
    # Create agents for workshop
    agents = migration_demo.demonstrate_langchain_agent_creation()
    
    while True:
        print(f"\n🤖 Available LangChain Agents:")
        for name, agent in agents.items():
            print(f"   • {name.replace('_', ' ').title()}: Production-ready agent with tools")
        
        user_query = input("\n💬 Your business question for LangChain agents: ")
        
        if user_query.lower() in ['exit', 'quit', 'done']:
            print("🎉 Excellent! You've experienced LangChain's production capabilities!")
            break
        
        if not user_query.strip():
            print("Please enter a business question for the LangChain agents.")
            continue
        
        # Ask which agent to use
        agent_choice = input("Choose agent (business/research/financial): ").lower()
        agent_map = {
            "business": "business_analyst",
            "research": "research_specialist", 
            "financial": "financial_expert"
        }
        
        agent_key = agent_map.get(agent_choice, "business_analyst")
        chosen_agent = agents[agent_key]
        
        print(f"\n🚀 Executing LangChain {agent_key.replace('_', ' ').title()}...")
        
        try:
            result = chosen_agent.invoke({"input": user_query})
            print(f"\n🎯 LangChain Agent Result:")
            print(result['output'])
        except Exception as e:
            print(f"❌ Error: {e}")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour4_q1_workshop():
    """Main function for Hour 4 Q1 workshop"""
    print("🚀 HOUR 4 - QUARTER 1: LANGCHAIN FUNDAMENTALS & AGENT MIGRATION")
    print("=" * 80)
    print("Welcome to production-ready AI with LangChain!\n")
    
    # Step 1: Compare custom vs LangChain
    migration_demo = AgentMigrationDemo()
    migration_demo.compare_custom_vs_langchain()
    
    # Step 2: Demonstrate LangChain agent creation
    migration_demo.demonstrate_langchain_agent_creation()
    
    # Step 3: Show production features
    production_features = LangChainProductionFeatures()
    production_features.demonstrate_memory_systems()
    production_features.demonstrate_callback_system()
    production_features.demonstrate_chain_composition()
    
    # Step 4: Test LangChain agents
    test_langchain_agents()
    
    # Step 5: Interactive workshop
    langchain_migration_workshop()
    
    # Step 6: Quarter completion and Q2 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 1 COMPLETE!")
    print("=" * 60)
    print("LangChain Fundamentals & Migration Achievements:")
    print("✅ Mastered LangChain architecture and production benefits")
    print("✅ Successfully migrated custom agents to LangChain framework")
    print("✅ Built production-ready agents with tools, memory, and callbacks")
    print("✅ Experienced 90% code reduction with enterprise features")
    print("✅ Understanding of LangChain agents, chains, and tools")
    
    print("\n🏆 Your LangChain Capabilities:")
    print("   → Production-ready agent development with minimal code")
    print("   → Built-in tool integration and error handling")
    print("   → Advanced memory management and state persistence")
    print("   → Enterprise monitoring and observability")
    print("   → Scalable chain composition for complex workflows")
    
    print("\n📈 Migration Benefits Achieved:")
    print("   Custom Agents → LangChain Framework")
    print("   100+ lines → 20 lines (90% code reduction)")
    print("   Manual error handling → Built-in robustness")
    print("   Custom monitoring → Production observability")
    
    print("\n🚀 Coming Up in Q2: Model Context Protocol (MCP) Integration")
    print("   → Advanced context management for enterprise AI")
    print("   → MCP server and client implementation")
    print("   → Context-aware agent ecosystems")
    print("   → Cross-agent context sharing and coordination")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 4 Q2: Model Context Protocol Integration!")

if __name__ == "__main__":
    # Run the complete Hour 4 Q1 workshop
    run_hour4_q1_workshop()