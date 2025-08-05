"""
Hour 2 - Quarter 1: Web Search Integration
==========================================

Learning Objectives:
- Add web search capabilities to your agent
- Learn intelligent query formulation
- Integrate real-time information with reasoning
- Build agents that can research and analyze current data

Duration: 15 minutes (after break)
Technical Skills: API integration, search query optimization, data synthesis
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

class WebSearchTool:
    """
    Professional web search integration using Tavily API
    Provides real-time information access for agents
    """
    
    def __init__(self):
        """Initialize the web search tool"""
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        
    def search(self, query, max_results=3):
        """
        Perform web search and return structured results
        """
        if not self.api_key:
            return {
                "success": False,
                "results": [],
                "error": "TAVILY_API_KEY not found. Please add it to your .env file.",
                "search_summary": "Search unavailable - API key missing"
            }
        
        try:
            # Prepare search request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "format": "json"
            }
            
            print(f"🔍 Searching for: {query}")
            
            # Execute search
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Process and format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.get("title", "No title"),
                        "content": result.get("content", "No content")[:500] + "...",
                        "url": result.get("url", "No URL"),
                        "score": result.get("score", 0)
                    })
                
                # Create summary
                summary = self._create_search_summary(query, formatted_results)
                
                return {
                    "success": True,
                    "results": formatted_results,
                    "search_summary": summary,
                    "query_used": query,
                    "total_results": len(formatted_results)
                }
            else:
                return {
                    "success": False,
                    "results": [],
                    "error": f"Search API error: {response.status_code}",
                    "search_summary": f"Search failed for: {query}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "results": [],
                "error": str(e),
                "search_summary": f"Search error for: {query}"
            }
    
    def _create_search_summary(self, query, results):
        """Create a concise summary of search results"""
        if not results:
            return f"No results found for: {query}"
        
        summary = f"Found {len(results)} results for '{query}':\n"
        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['title'][:60]}...\n"
        
        return summary

# =============================================================================
# CALCULATOR TOOL (from Hour 1)
# =============================================================================

def calculator_tool(expression):
    """Safe calculator tool from Hour 1"""
    try:
        allowed_names = {
            "__builtins__": None,
            "abs": abs, "round": round, "pow": pow,
            "max": max, "min": min
        }
        result = eval(expression.strip(), allowed_names, {})
        return {
            "success": True,
            "result": result,
            "formatted": f"{expression} = {result}"
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "formatted": f"Calculator Error: {e}"
        }

# =============================================================================
# WEB-ENABLED REASONING AGENT
# =============================================================================

class WebEnabledAgent:
    """
    Enhanced agent with web search capabilities
    Can research current information and integrate it with reasoning
    """
    
    def __init__(self):
        """Initialize the web-enabled agent"""
        load_dotenv()
        self.client = OpenAI()
        self.search_tool = WebSearchTool()
        
        # Enhanced prompt with web search awareness
        self.system_prompt = """You are an intelligent agent with web search and calculator capabilities.

AVAILABLE TOOLS:
1. WebSearch(query) - Search the internet for current information
2. Calculator(expression) - Perform precise mathematical calculations

TOOL USAGE STRATEGY:
- Use WebSearch when you need current information, market data, recent events, or facts you don't know
- Use Calculator for precise mathematical operations
- Combine both tools when analyzing current data with calculations
- Always explain why you're using each tool

FORMAT YOUR RESPONSE:
Thought: [Your reasoning about what information or calculation you need]
Action: [Either "WebSearch(query)" or "Calculator(expression)"]
Observation: [The result from the tool]
[Continue until you have enough information]

Final Answer: [Complete answer with sources when using web search]

SEARCH QUERY GUIDELINES:
- Make queries specific and relevant
- Use business/professional terms
- Include timeframes when needed (e.g., "2024", "current", "recent")
- Focus on factual, data-driven queries

Example:
Thought: I need to find current market information about electric vehicle sales.
Action: WebSearch(electric vehicle sales growth 2024 market data)
Observation: [Search results about EV market trends]

Thought: Now I need to calculate the growth percentage based on this data.
Action: Calculator((new_value - old_value) / old_value * 100)
Observation: Calculator result: growth percentage

Final Answer: Based on current market data, electric vehicle sales have grown by X% in 2024...
"""
    
    def research_and_solve(self, user_question, max_steps=10):
        """
        Solve problems using web research and calculations
        """
        print(f"\n🤖 Web-Enabled Agent received: {user_question}")
        print("🌐 Analyzing if web research is needed...\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]
        
        step_count = 0
        search_count = 0
        calc_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"🔄 Step {step_count}:")
            
            try:
                # Get agent response
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                
                # Process tool calls
                tool_result = None
                
                # Check for web search
                if "WebSearch(" in agent_response:
                    search_count += 1
                    query = self._extract_search_query(agent_response)
                    if query:
                        search_result = self.search_tool.search(query)
                        tool_result = search_result["search_summary"]
                        print(f"\n🌐 Search Result: {tool_result}")
                
                # Check for calculator
                elif "Calculator(" in agent_response:
                    calc_count += 1
                    expression = self._extract_calculator_expression(agent_response)
                    if expression:
                        calc_result = calculator_tool(expression)
                        tool_result = calc_result["formatted"]
                        print(f"\n🔢 Calculator Result: {tool_result}")
                
                # Update conversation
                messages.append({"role": "assistant", "content": agent_response})
                if tool_result:
                    messages.append({"role": "user", "content": f"Tool Result: {tool_result}"})
                
                print("-" * 50)
                
                # Check for completion
                if "Final Answer:" in agent_response:
                    print(f"✅ Research complete! Used {search_count} searches, {calc_count} calculations")
                    return self._extract_final_answer(agent_response)
                
                # Continue if no final answer
                if not tool_result:
                    messages.append({
                        "role": "user",
                        "content": "Continue with research or calculation, or provide Final Answer."
                    })
                
            except Exception as e:
                print(f"❌ Error in step {step_count}: {e}")
                return f"Error occurred: {e}"
        
        print("⚠️ Reached maximum research steps")
        return "Research incomplete - reached step limit"
    
    def _extract_search_query(self, response):
        """Extract search query from agent response"""
        import re
        match = re.search(r'WebSearch\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
    def _extract_calculator_expression(self, response):
        """Extract calculator expression from agent response"""
        import re
        match = re.search(r'Calculator\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
    def _extract_final_answer(self, response):
        """Extract the final answer"""
        lines = response.split('\n')
        for line in lines:
            if line.startswith("Final Answer:"):
                return line.replace("Final Answer:", "").strip()
        return "Final answer format not found"

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_search_need():
    """Show why agents need web search capabilities"""
    print("🔍 WHY AGENTS NEED WEB SEARCH")
    print("=" * 60)
    
    limitations = [
        {
            "scenario": "Market Analysis",
            "without_search": "Agent: 'I don't have current market data'",
            "with_search": "Agent researches latest market trends and provides analysis",
            "business_impact": "Informed decision-making vs. outdated assumptions"
        },
        {
            "scenario": "Competitor Research", 
            "without_search": "Agent: 'I can't access competitor information'",
            "with_search": "Agent finds competitor pricing, features, and strategies",
            "business_impact": "Competitive advantage vs. blind strategy"
        },
        {
            "scenario": "Current Events Impact",
            "without_search": "Agent: 'I don't know recent developments'",
            "with_search": "Agent researches relevant news and assesses business impact",
            "business_impact": "Proactive planning vs. reactive responses"
        }
    ]
    
    for i, limit in enumerate(limitations, 1):
        print(f"\n📊 Scenario {i}: {limit['scenario']}")
        print(f"   ❌ Without Search: {limit['without_search']}")
        print(f"   ✅ With Search: {limit['with_search']}")
        print(f"   💼 Business Impact: {limit['business_impact']}")
    
    print("\n🎯 Key Insight: Web search transforms agents from calculators to researchers!")

def demonstrate_smart_querying():
    """Show intelligent search query formulation"""
    print("\n🧠 INTELLIGENT SEARCH QUERY FORMULATION")
    print("=" * 60)
    
    query_examples = [
        {
            "user_question": "How is Tesla doing financially?",
            "poor_query": "Tesla",
            "smart_query": "Tesla financial performance Q3 2024 revenue earnings",
            "why_better": "Specific, timely, focused on financial metrics"
        },
        {
            "user_question": "What's the market size for our product?",
            "poor_query": "market size",
            "smart_query": "cloud storage market size 2024 growth forecast",
            "why_better": "Industry-specific with current timeframe"
        },
        {
            "user_question": "Are our prices competitive?",
            "poor_query": "competitive prices",
            "smart_query": "enterprise software pricing comparison 2024 SaaS",
            "why_better": "Category-specific with relevant context"
        }
    ]
    
    print(f"{'User Question':<30} {'Poor Query':<25} {'Smart Query':<35} {'Why Better':<30}")
    print("-" * 120)
    
    for example in query_examples:
        print(f"{example['user_question'][:29]:<30} {example['poor_query']:<25} {example['smart_query'][:34]:<35} {example['why_better'][:29]:<30}")
    
    print("\n🎯 Smart queries get better results and more relevant information!")

# =============================================================================
# TESTING WEB SEARCH CAPABILITIES
# =============================================================================

def test_web_search_integration():
    """Test the web-enabled agent with real-world scenarios"""
    print("\n🧪 TESTING WEB-ENABLED CAPABILITIES")
    print("=" * 60)
    
    agent = WebEnabledAgent()
    
    test_cases = [
        {
            "name": "Market Research Analysis",
            "question": "What's the current growth rate of the cloud computing market and what does this mean for our SaaS business?",
            "expected_tools": "Web search for market data + calculator for growth analysis"
        },
        {
            "name": "Competitive Intelligence",
            "question": "How does Microsoft's recent pricing compare to our current rates, and what should our pricing adjustment be?",
            "expected_tools": "Web search for Microsoft pricing + calculator for comparison"
        },
        {
            "name": "Economic Impact Analysis",
            "question": "How have recent inflation rates affected business software spending, and how should we adjust our sales forecasts?",
            "expected_tools": "Web search for economic data + calculator for forecast adjustments"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Web Search Test {i}: {test['name']}")
        print(f"🌐 Expected Tools: {test['expected_tools']}")
        print(f"❓ Question: {test['question']}")
        
        result = agent.research_and_solve(test['question'])
        print(f"🏆 Research Result: {result}")
        print("\n" + "=" * 80)
        
        if i < len(test_cases):
            input("Press Enter to continue to next research test...")

# =============================================================================
# CAPABILITY COMPARISON
# =============================================================================

def compare_tool_evolution():
    """Show the evolution from single tool to multi-tool"""
    print("\n📈 TOOL EVOLUTION: CALCULATOR → WEB SEARCH → MULTI-TOOL")
    print("=" * 70)
    
    evolution_stages = [
        {
            "stage": "Hour 1 Q4: Calculator Only",
            "capabilities": "Precise math, financial calculations",
            "limitations": "No access to current information",
            "use_case": "Internal calculations and analysis"
        },
        {
            "stage": "Hour 2 Q1: + Web Search",
            "capabilities": "Research + calculations, current data",
            "limitations": "Manual tool coordination",
            "use_case": "Market research with financial analysis"
        },
        {
            "stage": "Coming: Multi-Tool Coordination",
            "capabilities": "Intelligent tool chaining, file processing",
            "limitations": "Individual agent only",
            "use_case": "Complete business intelligence workflows"
        }
    ]
    
    for stage in evolution_stages:
        print(f"\n🔧 {stage['stage']}")
        print(f"   ✅ Capabilities: {stage['capabilities']}")
        print(f"   ⚠️ Limitations: {stage['limitations']}")
        print(f"   💼 Use Case: {stage['use_case']}")
    
    print("\n🎯 Each tool addition exponentially increases agent capabilities!")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def web_search_workshop():
    """Interactive workshop with web-enabled agent"""
    print("\n🎯 WEB SEARCH INTEGRATION WORKSHOP")
    print("=" * 60)
    
    agent = WebEnabledAgent()
    
    print("Test your web-enabled agent with real business research questions!")
    print("Great challenges for web search:")
    print("• Market trends and competitor analysis")
    print("• Current pricing and financial data")
    print("• Industry news and economic impacts")
    print("• Technology trends and business implications")
    print("\nType 'exit' to finish this quarter.")
    
    while True:
        user_question = input("\n💬 Your research question: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Excellent! Your agent can now research and analyze real-world information!")
            break
        
        if user_question.strip():
            result = agent.research_and_solve(user_question)
            print(f"\n🎯 Research Result: {result}")
        else:
            print("Please enter a research question that requires current information.")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour2_q1_workshop():
    """Main function for Hour 2 Q1 workshop"""
    print("🚀 HOUR 2 - QUARTER 1: WEB SEARCH INTEGRATION")
    print("=" * 70)
    print("Welcome back from your break! Ready to give your agents internet access?\n")
    
    # Step 1: Show why search is needed
    demonstrate_search_need()
    
    # Step 2: Show smart querying
    demonstrate_smart_querying()
    
    # Step 3: Test web search capabilities
    test_web_search_integration()
    
    # Step 4: Show tool evolution
    compare_tool_evolution()
    
    # Step 5: Interactive workshop
    web_search_workshop()
    
    # Step 6: Quarter completion and Q2 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 1 COMPLETE!")
    print("=" * 60)
    print("Web Search Integration Achievements:")
    print("✅ Real-time information access capabilities")
    print("✅ Intelligent search query formulation")
    print("✅ Data synthesis and analysis integration")
    print("✅ Current market and competitor research")
    print("✅ Foundation for comprehensive business intelligence")
    
    print("\n🏆 Your Agent Can Now:")
    print("   → Research current market trends and data")
    print("   → Analyze competitor information and pricing")
    print("   → Combine web research with precise calculations")
    print("   → Provide data-driven business recommendations")
    
    print("\n🚀 Coming Up in Q2: File Processing Capability")
    print("   → Add document analysis and summarization")
    print("   → Process reports, contracts, and business documents")
    print("   → Combine file analysis with web research")
    print("   → Create comprehensive document intelligence")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 2 Q2: File Processing Integration!")

if __name__ == "__main__":
    # Run the complete Hour 2 Q1 workshop
    run_hour2_q1_workshop()