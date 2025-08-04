"""
Hour 1 - Quarter 3: Enhanced Reasoning & Error Handling
=======================================================

Learning Objectives:
- Add self-correction capabilities to your agent
- Implement retry logic and error recovery
- Create production-ready robustness
- Build agents that verify their own work

Duration: 15 minutes
Technical Skills: Error handling, self-correction, validation, retry logic
"""

import os
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# ENHANCED REASONING AGENT CLASS
# =============================================================================

class EnhancedReasoningAgent:
    """
    Enhanced version of our basic agent with:
    - Self-correction capabilities
    - Retry logic for failed attempts
    - Better calculation validation
    - Robust error handling
    - Production-ready reliability
    """
    
    def __init__(self):
        """Initialize the enhanced agent"""
        load_dotenv()
        self.client = OpenAI()
        self.max_retries = 3
        self.validation_enabled = True
        
        # Enhanced ReAct Prompt with self-correction instructions
        self.system_prompt = """You are an enhanced reasoning agent with self-correction capabilities.

CRITICAL REQUIREMENTS:
1. Follow the ReAct pattern: Thought → Action → Observation
2. ALWAYS show explicit calculations with numbers
3. VERIFY your math in a separate step
4. If you find an error, CORRECT it immediately
5. Be precise and show your work clearly

Format your response EXACTLY like this:

Thought: [Your reasoning about what to do next]
Action: [Explicit calculation with actual numbers, e.g., "Calculate: 500 × 6 = 3,000"]
Observation: [The result and what it means]

For math problems, ALWAYS include a verification step:
Thought: Let me verify this calculation is correct.
Action: Verify: [show the reverse calculation or alternative method]
Observation: [confirm if correct or identify errors]

If you find an error, immediately correct it:
Thought: I found an error in my calculation. Let me fix it.
Action: Correct calculation: [show the right math]
Observation: [the corrected result]

ALWAYS end with: Final Answer: [complete, verified answer]

Example:
Thought: I need to calculate 15% tip on $47.83.
Action: Calculate: 47.83 × 0.15 = 7.1745
Observation: This gives us $7.1745, but I should round to cents.

Thought: Let me round to proper currency format.
Action: Round: 7.1745 → $7.17
Observation: $7.17 is the properly formatted tip amount.

Thought: Let me verify this calculation is correct.
Action: Verify: 7.17 ÷ 47.83 = 0.1499 ≈ 0.15 (15%)
Observation: Verification confirms this is correct.

Final Answer: The 15% tip on $47.83 is $7.17.
"""
    
    def enhanced_reasoning(self, user_question, max_steps=7):
        """
        Enhanced reasoning with retry logic and self-correction
        """
        print(f"\n🤖 Enhanced Agent received: {user_question}")
        print("🧠 Thinking with enhanced reasoning and self-correction...\n")
        
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_reasoning(user_question, max_steps, attempt + 1)
                if result and "error" not in result.lower():
                    return result
                else:
                    print(f"⚠️ Attempt {attempt + 1} had issues, retrying...\n")
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print("🔄 Retrying with enhanced error handling...\n")
                    time.sleep(1)
        
        return "Could not complete reasoning after multiple attempts"
    
    def _attempt_reasoning(self, user_question, max_steps, attempt_number):
        """Single reasoning attempt with enhanced logic"""
        
        print(f"🔄 Reasoning Attempt {attempt_number}")
        print("-" * 40)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]
        
        step_count = 0
        calculation_errors = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"📍 Step {step_count}:")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=400
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                
                # Validate mathematical calculations in the response
                if self.validation_enabled:
                    validation_result = self._validate_calculations(agent_response)
                    if not validation_result["valid"]:
                        calculation_errors += 1
                        print(f"\n⚠️ Calculation Error Detected: {validation_result['error']}")
                        
                        if calculation_errors < 2:  # Allow one correction attempt
                            correction_prompt = f"I found an error in your calculation: {validation_result['error']}. Please correct this and continue."
                            messages.append({"role": "assistant", "content": agent_response})
                            messages.append({"role": "user", "content": correction_prompt})
                            print("🔧 Requesting self-correction...\n")
                            continue
                
                print("-" * 50)
                messages.append({"role": "assistant", "content": agent_response})
                
                # Check for completion
                if "Final Answer:" in agent_response:
                    print("✅ Enhanced reasoning complete with verification!")
                    return self._extract_final_answer(agent_response)
                
                # Continue reasoning if needed
                messages.append({
                    "role": "user", 
                    "content": "Continue with verification step if needed, or provide Final Answer."
                })
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in reasoning step {step_count}: {e}")
                return f"Error in step {step_count}: {e}"
        
        print("⚠️ Reached maximum reasoning steps")
        return "Reasoning incomplete - reached step limit"
    
    def _validate_calculations(self, response):
        """
        Validate mathematical calculations in the agent's response
        This is our self-correction mechanism
        """
        # Look for calculation patterns like "47.83 × 0.15 = 7.1745"
        calc_pattern = r'(\d+\.?\d*)\s*[×*]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        calculations = re.findall(calc_pattern, response)
        
        for calc in calculations:
            try:
                num1, num2, claimed_result = float(calc[0]), float(calc[1]), float(calc[2])
                actual_result = num1 * num2
                
                # Allow small floating point differences
                if abs(actual_result - claimed_result) > 0.01:
                    return {
                        "valid": False,
                        "error": f"{calc[0]} × {calc[1]} should equal {actual_result:.4f}, not {claimed_result}"
                    }
            except (ValueError, IndexError):
                continue
        
        return {"valid": True, "error": None}
    
    def _extract_final_answer(self, response):
        """Extract and clean the final answer"""
        lines = response.split('\n')
        for line in lines:
            if line.startswith("Final Answer:"):
                return line.replace("Final Answer:", "").strip()
        return "Final answer format not found"

# =============================================================================
# PROBLEM IDENTIFICATION DEMO
# =============================================================================

def demonstrate_problems_with_basic_agent():
    """Show the problems we're solving with enhancement"""
    print("🔍 IDENTIFYING PROBLEMS WITH BASIC AGENTS")
    print("=" * 60)
    
    problems = [
        {
            "problem": "Calculation Errors",
            "example": "Agent says '15% of $100 = $16' (should be $15)",
            "impact": "Wrong business decisions, lost money"
        },
        {
            "problem": "No Self-Verification",
            "example": "Agent doesn't double-check its math",
            "impact": "Errors go undetected and propagate"
        },
        {
            "problem": "Poor Error Recovery",
            "example": "Agent gets confused and gives up",
            "impact": "Unreliable in production environments"
        },
        {
            "problem": "Vague Calculations", 
            "example": "Agent says 'after calculating...' without showing work",
            "impact": "Can't verify or debug the reasoning"
        }
    ]
    
    for i, issue in enumerate(problems, 1):
        print(f"\n❌ Problem {i}: {issue['problem']}")
        print(f"   Example: {issue['example']}")
        print(f"   Business Impact: {issue['impact']}")
    
    print("\n🎯 Solution: Enhanced Agent with Self-Correction!")

def demonstrate_enhancements():
    """Show what enhancements we're adding"""
    print("\n🔧 ENHANCED AGENT CAPABILITIES")
    print("=" * 60)
    
    enhancements = [
        {
            "feature": "Self-Correction",
            "description": "Agent verifies its own calculations",
            "benefit": "Catches and fixes errors automatically"
        },
        {
            "feature": "Retry Logic",
            "description": "Multiple attempts if reasoning fails",
            "benefit": "More reliable in edge cases"
        },
        {
            "feature": "Explicit Calculations",
            "description": "Shows actual numbers: '500 × 6 = 3,000'",
            "benefit": "Transparent, verifiable reasoning"
        },
        {
            "feature": "Validation System",
            "description": "Automated checking of mathematical results",
            "benefit": "Production-ready accuracy"
        }
    ]
    
    for enhancement in enhancements:
        print(f"\n✅ {enhancement['feature']}")
        print(f"   How: {enhancement['description']}")
        print(f"   Benefit: {enhancement['benefit']}")

# =============================================================================
# TESTING ENHANCED CAPABILITIES
# =============================================================================

def test_enhanced_capabilities():
    """Test the enhanced agent with challenging problems"""
    print("\n🧪 TESTING ENHANCED CAPABILITIES")
    print("=" * 60)
    
    agent = EnhancedReasoningAgent()
    
    # Test cases that would challenge a basic agent
    test_cases = [
        {
            "name": "Complex Percentage Chain",
            "question": "If a product costs $80, we mark it up 25%, then give a 10% discount, what's the final price?",
            "challenge": "Multi-step percentages with potential rounding issues"
        },
        {
            "name": "Budget Allocation Verification",
            "question": "We have $10,000 budget. After spending 35% on marketing, 28% on development, and $1,500 on overhead, how much remains?",
            "challenge": "Mixed percentages and fixed amounts, requires verification"
        },
        {
            "name": "ROI Calculation",
            "question": "We invested $25,000 and earned $31,250 back. What's our ROI percentage?", 
            "challenge": "Requires precise calculation and percentage conversion"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Enhanced Test {i}: {test['name']}")
        print(f"🎯 Challenge: {test['challenge']}")
        print(f"❓ Question: {test['question']}")
        
        result = agent.enhanced_reasoning(test['question'])
        print(f"🏆 Enhanced Result: {result}")
        print("\n" + "=" * 80)
        
        if i < len(test_cases):
            input("Press Enter to continue to next enhanced test...")

# =============================================================================
# COMPARISON DEMONSTRATION
# =============================================================================

def compare_basic_vs_enhanced():
    """Side-by-side comparison of capabilities"""
    print("\n⚖️ BASIC vs ENHANCED AGENT COMPARISON")
    print("=" * 60)
    
    comparison = [
        ("Error Handling", "Basic retry only", "Multi-level retry + self-correction"),
        ("Calculations", "May be vague", "Always explicit with numbers"),
        ("Verification", "None", "Automatic validation + correction"),
        ("Reliability", "Good for simple tasks", "Production-ready robustness"),
        ("Debugging", "Hard to trace errors", "Clear error detection + fixing"),
        ("Business Use", "Prototype/demo", "Mission-critical applications")
    ]
    
    print(f"{'Capability':<15} {'Basic Agent':<25} {'Enhanced Agent':<30}")
    print("-" * 70)
    
    for capability, basic, enhanced in comparison:
        print(f"{capability:<15} {basic:<25} {enhanced:<30}")
    
    print(f"\n🎯 Key Insight: Enhanced agents are ready for real business use!")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def enhanced_workshop_challenge():
    """Interactive challenge with enhanced agent"""
    print("\n🎯 ENHANCED AGENT WORKSHOP CHALLENGE")
    print("=" * 60)
    
    agent = EnhancedReasoningAgent()
    
    print("Test your enhanced agent with complex business scenarios!")
    print("Try problems that might challenge or confuse a basic agent:")
    print("\nSuggested challenges:")
    print("• Multi-step financial calculations")
    print("• Complex percentage problems")
    print("• ROI and profit margin analysis")
    print("• Budget allocations with mixed units")
    print("\nType 'exit' to finish.")
    
    while True:
        user_question = input("\n💬 Your challenging question: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Excellent! You've built a production-ready reasoning agent!")
            break
        
        if user_question.strip():
            result = agent.enhanced_reasoning(user_question)
            print(f"\n🎯 Enhanced Agent Result: {result}")
        else:
            print("Please enter a challenging question for the enhanced agent.")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour1_q3_workshop():
    """Main function for Hour 1 Q3 workshop"""
    print("🚀 HOUR 1 - QUARTER 3: ENHANCED REASONING & ERROR HANDLING")
    print("=" * 70)
    
    # Step 1: Identify problems we're solving
    demonstrate_problems_with_basic_agent()
    
    # Step 2: Show our enhancements
    demonstrate_enhancements()
    
    # Step 3: Test enhanced capabilities
    test_enhanced_capabilities()
    
    # Step 4: Compare basic vs enhanced
    compare_basic_vs_enhanced()
    
    # Step 5: Interactive challenge
    enhanced_workshop_challenge()
    
    # Step 6: Wrap up and prepare for Q4
    print("\n" + "=" * 60)
    print("🎉 QUARTER 3 COMPLETE!")
    print("=" * 60)
    print("Enhanced Agent Achievements:")
    print("✅ Self-correction and verification capabilities")
    print("✅ Robust error handling and retry logic")
    print("✅ Explicit calculation display")
    print("✅ Production-ready reliability")
    print("✅ Automated validation systems")
    
    print("\n🏆 Your Agent Evolution:")
    print("   Q2: Basic reasoning agent")
    print("   Q3: Production-ready enhanced agent")
    print("   Q4: Multi-tool integration coming next!")
    
    print("\n🚀 Coming Up in Q4: Basic Tool Integration")
    print("   → Add calculator tool for perfect math")
    print("   → Intelligent tool selection logic") 
    print("   → Foundation for multi-tool agents")
    print("   → Real-world problem solving capabilities")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 1 Q4 - Tool Integration!")

if __name__ == "__main__":
    # Run the complete Hour 1 Q3 workshop
    run_hour1_q3_workshop()