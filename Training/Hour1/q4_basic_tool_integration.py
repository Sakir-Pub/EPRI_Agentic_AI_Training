"""
Hour 1 - Quarter 4: Basic Tool Integration
==========================================

Learning Objectives:
- Add calculator tool for perfect mathematical accuracy
- Implement intelligent tool selection logic
- Create foundation for multi-tool agent systems
- Build agents that know when and how to use tools

Duration: 15 minutes
Technical Skills: Tool integration, function calling, tool selection logic
"""

import os
import re
import math
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def calculator_tool(expression):
    """
    Safe calculator tool for mathematical operations.
    Supports +, -, *, /, %, **, sqrt, round, etc.
    """
    try:
        # Safety: Only allow mathematical operations
        allowed_names = {
            "__builtins__": None,
            "abs": abs,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "ceil": math.ceil,
            "floor": math.floor,
            "max": max,
            "min": min
        }
        
        # Clean the expression
        clean_expr = expression.strip()
        
        # Evaluate safely
        result = eval(clean_expr, allowed_names, {})
        
        return {
            "success": True,
            "result": result,
            "formatted": f"{expression} = {result}",
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "formatted": f"Error calculating {expression}",
            "error": str(e)
        }

def format_currency(amount):
    """Format numbers as currency"""
    try:
        return f"${amount:,.2f}"
    except:
        return str(amount)

def format_percentage(decimal_value):
    """Convert decimal to percentage"""
    try:
        return f"{decimal_value * 100:.2f}%"
    except:
        return str(decimal_value)

# =============================================================================
# TOOL-ENABLED REASONING AGENT
# =============================================================================

class ToolEnabledAgent:
    """
    Advanced agent that can use tools for accurate calculations.
    Implements intelligent tool selection - knows when to use tools vs. reasoning.
    """
    
    def __init__(self):
        """Initialize the tool-enabled agent"""
        load_dotenv()
        self.client = OpenAI()
        self.available_tools = {
            "calculator": calculator_tool,
            "format_currency": format_currency,
            "format_percentage": format_percentage
        }
        
        # Enhanced prompt with tool awareness
        self.system_prompt = """You are an intelligent agent with access to tools for accurate calculations.

AVAILABLE TOOLS:
1. Calculator(expression) - For precise mathematical calculations
2. FormatCurrency(amount) - Format numbers as currency ($X.XX)
3. FormatPercentage(decimal) - Convert decimals to percentages (X.XX%)

TOOL USAGE RULES:
- Use Calculator for ANY mathematical operation to ensure accuracy
- Use tools when you need precise calculations
- Always show what tool you're using and why
- Verify complex calculations with multiple tool calls if needed

FORMAT YOUR RESPONSE:
Thought: [Your reasoning about what to do]
Action: [Either "Calculate using Calculator(expression)" or "Reason: explanation"]
Observation: [The result from the tool or your reasoning]

For tool usage, write exactly:
Action: Calculator(47.83 * 0.15)
[System will provide the tool result]

Continue until you reach a Final Answer.

Example with tools:
Thought: I need to calculate 15% tip on $47.83 precisely.
Action: Calculator(47.83 * 0.15)
Observation: Calculator result: 47.83 * 0.15 = 7.1745

Thought: I should round this to proper currency format.
Action: Calculator(round(7.1745, 2))
Observation: Calculator result: round(7.1745, 2) = 7.17

Thought: Let me verify this is exactly 15%.
Action: Calculator(7.17 / 47.83)
Observation: Calculator result: 7.17 / 47.83 = 0.14995817...

Final Answer: The 15% tip on $47.83 is $7.17.
"""
    
    def solve_with_tools(self, user_question, max_steps=8):
        """
        Solve problems using both reasoning and tools intelligently
        """
        print(f"\n🤖 Tool-Enabled Agent received: {user_question}")
        print("🛠️ Analyzing if tools are needed for accurate solution...\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]
        
        step_count = 0
        tool_usage_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"🔄 Step {step_count}:")
            
            try:
                # Get agent response
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=400
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                
                # Check for tool usage
                tool_result = self._process_tool_calls(agent_response)
                if tool_result:
                    tool_usage_count += 1
                    print(f"\n🛠️ Tool Result: {tool_result}")
                    messages.append({"role": "assistant", "content": agent_response})
                    messages.append({"role": "user", "content": f"Tool Result: {tool_result}"})
                else:
                    messages.append({"role": "assistant", "content": agent_response})
                
                print("-" * 50)
                
                # Check for completion
                if "Final Answer:" in agent_response:
                    print(f"✅ Problem solved using {tool_usage_count} tool calls!")
                    return self._extract_final_answer(agent_response)
                
                # Continue if needed
                if "Final Answer:" not in agent_response:
                    messages.append({
                        "role": "user",
                        "content": "Continue with next step or provide Final Answer."
                    })
                
            except Exception as e:
                print(f"❌ Error in step {step_count}: {e}")
                return f"Error occurred: {e}"
        
        print("⚠️ Reached maximum steps")
        return "Solution incomplete - reached step limit"
    
    def _process_tool_calls(self, response):
        """
        Process tool calls in the agent's response
        """
        # Look for Calculator calls
        calc_pattern = r'Calculator\((.*?)\)'
        calc_matches = re.findall(calc_pattern, response)
        
        if calc_matches:
            expression = calc_matches[0]
            result = calculator_tool(expression)
            if result["success"]:
                return result["formatted"]
            else:
                return f"Calculator Error: {result['error']}"
        
        # Look for currency formatting
        currency_pattern = r'FormatCurrency\((.*?)\)'
        currency_matches = re.findall(currency_pattern, response)
        
        if currency_matches:
            try:
                amount = float(currency_matches[0])
                return format_currency(amount)
            except:
                return "Currency formatting error"
        
        # Look for percentage formatting
        percent_pattern = r'FormatPercentage\((.*?)\)'
        percent_matches = re.findall(percent_pattern, response)
        
        if percent_matches:
            try:
                decimal = float(percent_matches[0])
                return format_percentage(decimal)
            except:
                return "Percentage formatting error"
        
        return None
    
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

def demonstrate_tool_advantage():
    """Show why tools make agents more reliable"""
    print("🔍 WHY TOOLS MAKE AGENTS MORE RELIABLE")
    print("=" * 60)
    
    comparisons = [
        {
            "scenario": "Calculating 15% tip on $47.83",
            "without_tools": "Agent estimates: 'About $7.18'",
            "with_tools": "Calculator(47.83 * 0.15) = 7.1745 → $7.17",
            "impact": "Precise vs. approximate results"
        },
        {
            "scenario": "Complex ROI calculation",
            "without_tools": "Agent might make rounding errors",
            "with_tools": "Multiple calculator calls ensure accuracy",
            "impact": "Critical financial decisions require precision"
        },
        {
            "scenario": "Multi-step budget calculations",
            "without_tools": "Errors compound through steps",
            "with_tools": "Each step verified with calculator",
            "impact": "Reliable business planning"
        }
    ]
    
    for i, comp in enumerate(comparisons, 1):
        print(f"\n📊 Scenario {i}: {comp['scenario']}")
        print(f"   ❌ Without Tools: {comp['without_tools']}")
        print(f"   ✅ With Tools: {comp['with_tools']}")
        print(f"   💼 Business Impact: {comp['impact']}")
    
    print("\n🎯 Key Insight: Tools provide precision, agents provide intelligence!")

def demonstrate_tool_selection_logic():
    """Show how agents decide when to use tools"""
    print("\n🧠 INTELLIGENT TOOL SELECTION")
    print("=" * 60)
    
    decision_matrix = [
        ("Simple conceptual question", "What is ROI?", "Reasoning only"),
        ("Rough estimation", "About how much is 20% of $100?", "Reasoning only"),
        ("Precise calculation", "Calculate exact 15.25% of $1,247.83", "Calculator tool"),
        ("Multi-step math", "Compound interest over 5 years", "Multiple calculator calls"),
        ("Formatting output", "Display as currency/percentage", "Formatting tools")
    ]
    
    print(f"{'Question Type':<20} {'Example':<35} {'Tool Strategy':<25}")
    print("-" * 80)
    
    for q_type, example, strategy in decision_matrix:
        print(f"{q_type:<20} {example:<35} {strategy:<25}")
    
    print("\n🎯 Agent learns to choose the right approach for each situation!")

# =============================================================================
# TESTING TOOL CAPABILITIES
# =============================================================================

def test_tool_integration():
    """Test the tool-enabled agent with challenging problems"""
    print("\n🧪 TESTING TOOL-ENABLED CAPABILITIES")
    print("=" * 60)
    
    agent = ToolEnabledAgent()
    
    test_cases = [
        {
            "name": "Precise Financial Calculation",
            "question": "If I invest $1,247.83 at 4.25% annual interest compounded monthly for 3 years, what's the final amount?",
            "tools_needed": "Multiple calculator calls for compound interest formula"
        },
        {
            "name": "Complex Business Scenario",
            "question": "Our quarterly revenue is $847,329. We spend 23.7% on salaries, 18.4% on marketing, $125,000 on rent, and 8.9% on R&D. What's our profit margin?",
            "tools_needed": "Percentage calculations, currency formatting, profit calculation"
        },
        {
            "name": "Multi-Step ROI Analysis",
            "question": "We bought equipment for $75,000, it generates $2,300 monthly revenue, and costs $680 monthly to operate. What's the ROI after 18 months?",
            "tools_needed": "Revenue calculation, cost calculation, ROI formula, percentage formatting"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Tool Test {i}: {test['name']}")
        print(f"🛠️ Expected Tools: {test['tools_needed']}")
        print(f"❓ Question: {test['question']}")
        
        result = agent.solve_with_tools(test['question'])
        print(f"🏆 Tool-Enhanced Result: {result}")
        print("\n" + "=" * 80)
        
        if i < len(test_cases):
            input("Press Enter to continue to next tool test...")

# =============================================================================
# CAPABILITY COMPARISON
# =============================================================================

def compare_agent_evolution():
    """Show the complete evolution from Q1 to Q4"""
    print("\n📈 YOUR AGENT EVOLUTION - HOUR 1 COMPLETE")
    print("=" * 60)
    
    evolution = [
        ("Q1 - Concept", "Understanding ReAct and Agentic AI", "Knowledge"),
        ("Q2 - Basic Agent", "Simple reasoning, step-by-step thinking", "Basic functionality"),
        ("Q3 - Enhanced Agent", "Self-correction, error handling, verification", "Production reliability"),
        ("Q4 - Tool-Enabled", "Calculator integration, tool selection", "Practical capabilities")
    ]
    
    print(f"{'Quarter':<20} {'Capability':<40} {'Achievement':<20}")
    print("-" * 80)
    
    for quarter, capability, achievement in evolution:
        print(f"{quarter:<20} {capability:<40} {achievement:<20}")
    
    capabilities_gained = [
        "✅ ReAct pattern implementation",
        "✅ Step-by-step reasoning logic", 
        "✅ Self-correction and verification",
        "✅ Robust error handling",
        "✅ Tool integration and selection",
        "✅ Precise mathematical calculations",
        "✅ Production-ready reliability",
        "✅ Foundation for multi-tool systems"
    ]
    
    print(f"\n🏆 Complete Capabilities Gained:")
    for capability in capabilities_gained:
        print(f"   {capability}")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def tool_integration_workshop():
    """Interactive workshop with tool-enabled agent"""
    print("\n🎯 TOOL INTEGRATION WORKSHOP CHALLENGE")
    print("=" * 60)
    
    agent = ToolEnabledAgent()
    
    print("Test your tool-enabled agent with precision-required problems!")
    print("Great challenges for tool usage:")
    print("• Complex financial calculations")
    print("• Multi-step business scenarios")
    print("• Precise percentage and currency problems")
    print("• Compound calculations requiring accuracy")
    print("\nType 'exit' to finish Hour 1.")
    
    while True:
        user_question = input("\n💬 Your precision-required question: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Outstanding! You've completed Hour 1 with a full tool-enabled agent!")
            break
        
        if user_question.strip():
            result = agent.solve_with_tools(user_question)
            print(f"\n🎯 Tool-Enhanced Result: {result}")
        else:
            print("Please enter a question requiring precise calculations.")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour1_q4_workshop():
    """Main function for Hour 1 Q4 workshop"""
    print("🚀 HOUR 1 - QUARTER 4: BASIC TOOL INTEGRATION")
    print("=" * 70)
    
    # Step 1: Show tool advantages
    demonstrate_tool_advantage()
    
    # Step 2: Explain tool selection logic
    demonstrate_tool_selection_logic()
    
    # Step 3: Test tool capabilities
    test_tool_integration()
    
    # Step 4: Show complete evolution
    compare_agent_evolution()
    
    # Step 5: Interactive workshop
    tool_integration_workshop()
    
    # Step 6: Hour 1 completion and Hour 2 preview
    print("\n" + "=" * 60)
    print("🎉 HOUR 1 COMPLETE - CONGRATULATIONS!")
    print("=" * 60)
    print("Your Amazing Journey:")
    print("✅ Q1: Understood Agentic AI concepts and ReAct pattern")
    print("✅ Q2: Built your first reasoning agent")
    print("✅ Q3: Enhanced with self-correction and error handling")
    print("✅ Q4: Added tool integration for precision and reliability")
    print()
    print("🏆 You now have a production-ready, tool-enabled agent!")
    print("   → Handles complex business calculations")
    print("   → Self-corrects and verifies results")
    print("   → Uses tools intelligently for precision")
    print("   → Ready for real-world applications")
    
    print("\n" + "=" * 60)
    print("🚀 COMING UP IN HOUR 2: ENHANCED MULTI-TOOL AGENT")
    print("=" * 60)
    print("Next Level Capabilities:")
    print("🌐 Web Search Integration - Real-time information gathering")
    print("📄 File Processing - Document analysis and summarization") 
    print("🔗 Multi-Tool Coordination - Intelligent tool chaining")
    print("🧠 Advanced Reasoning - Complex workflow planning")
    print()
    print("Hour 2 Preview: Your agent will research competitors online,")
    print("analyze documents, and create comprehensive business reports!")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Take a break! Ready for Hour 2 when you return!")
    print("🎯 You've mastered the foundations - now for the advanced capabilities!")

if __name__ == "__main__":
    # Run the complete Hour 1 Q4 workshop
    run_hour1_q4_workshop()