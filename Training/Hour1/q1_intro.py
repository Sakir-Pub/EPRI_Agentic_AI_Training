"""
Hour 1 - Quarter 1: Agentic AI Introduction & Demo
=================================================

Learning Objectives:
- Understand the evolution from Traditional → Generative → Agentic AI
- Learn ReAct (Reason-Act-Observe) pattern fundamentals
- Set up basic development environment
- See the difference between AI paradigms in action

Duration: 15 minutes
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# =============================================================================
# SETUP & ENVIRONMENT CONFIGURATION
# =============================================================================

def setup_environment():
    """
    Set up the basic environment for our Agentic AI tutorial.
    This includes loading API keys and checking dependencies.
    """
    print("🔧 Setting up your Agentic AI development environment...")
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in .env file")
        print("📝 Please create a .env file with your OpenAI API key")
        return False
    
    print("✅ Environment setup complete!")
    print("🚀 Ready to build Agentic AI systems!")
    return True

# =============================================================================
# AI EVOLUTION DEMONSTRATION
# =============================================================================

def demonstrate_traditional_ai():
    """
    Traditional AI: Rule-based, deterministic responses
    Limited to predefined scenarios
    """
    print("\n" + "="*60)
    print("🔹 TRADITIONAL AI DEMONSTRATION")
    print("="*60)
    
    # Simulate traditional rule-based AI
    def traditional_ai_calculator(operation, a, b):
        """Traditional AI: Fixed rules, no reasoning"""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        else:
            return "Error: Operation not supported"
    
    print("📝 Task: Calculate 15% tip on $47.83")
    print("🤖 Traditional AI Response:")
    print("   Error: 'calculate tip' operation not supported")
    print("   → Can only do: add, multiply")
    print("   → No reasoning capability")
    print("   → Requires exact command match")

def demonstrate_generative_ai():
    """
    Generative AI: Can create content but lacks reasoning
    Better than traditional but still limited
    """
    print("\n" + "="*60)
    print("🔸 GENERATIVE AI DEMONSTRATION")
    print("="*60)
    
    print("📝 Task: Calculate 15% tip on $47.83")
    print("🤖 Generative AI Response:")
    print("   'To calculate a 15% tip on $47.83, multiply by 0.15'")
    print("   'Result: $7.17'")
    print("   → Can generate relevant content")
    print("   → May make calculation errors")
    print("   → No self-verification")

def demonstrate_agentic_ai():
    """
    Agentic AI: Reasons, acts, observes, and corrects
    Shows the ReAct pattern in action
    """
    print("\n" + "="*60)
    print("🔺 AGENTIC AI DEMONSTRATION")
    print("="*60)
    
    print("📝 Task: Calculate 15% tip on $47.83")
    print("🤖 Agentic AI Response:")
    print()
    
    # Simulate the ReAct pattern
    steps = [
        {
            "thought": "I need to calculate 15% of $47.83. Let me break this down step by step.",
            "action": "Calculate 47.83 × 0.15",
            "observation": "Result: 7.1745"
        },
        {
            "thought": "The result is 7.1745, but for money I should round to 2 decimal places.",
            "action": "Round 7.1745 to 2 decimal places",
            "observation": "Result: $7.17"
        },
        {
            "thought": "Let me verify this is correct by checking: $7.17 / $47.83 = 0.15 (15%)",
            "action": "Verify: 7.17 ÷ 47.83",
            "observation": "Result: 0.1499... ≈ 0.15 ✓"
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"   Step {i}:")
        print(f"   💭 Thought: {step['thought']}")
        print(f"   🎯 Action: {step['action']}")
        print(f"   👁️  Observation: {step['observation']}")
        print()
        time.sleep(1)  # Pause for dramatic effect
    
    print("   ✅ Final Answer: The 15% tip on $47.83 is $7.17")
    print()
    print("   🧠 Key Capabilities Demonstrated:")
    print("   → Step-by-step reasoning")
    print("   → Self-correction and verification")
    print("   → Appropriate rounding for context")
    print("   → Quality assurance")

# =============================================================================
# REACT PATTERN EXPLANATION
# =============================================================================

def explain_react_pattern():
    """
    Explain the fundamental ReAct (Reason-Act-Observe) pattern
    that powers Agentic AI systems
    """
    print("\n" + "="*60)
    print("🧠 THE ReAct PATTERN - CORE OF AGENTIC AI")
    print("="*60)
    
    react_cycle = """
    ┌─────────────────────────────────────────────────────────┐
    │                    ReAct Cycle                          │
    └─────────────────────────────────────────────────────────┘
    
    1. REASON (Think) 💭
       │ "What do I need to do?"
       │ "What information do I have?"
       │ "What's my next step?"
       ▼
    
    2. ACT (Do) 🎯
       │ Execute a specific action
       │ Use a tool or make a calculation
       │ Gather information
       ▼
    
    3. OBSERVE (Learn) 👁️
       │ Analyze the result
       │ Check if it's correct
       │ Decide next steps
       ▼
    
    4. REPEAT until goal achieved ✅
    """
    
    print(react_cycle)
    
    print("\n🔑 Why ReAct is Revolutionary:")
    print("   • Mimics human problem-solving")
    print("   • Self-correcting and adaptive")
    print("   • Can handle complex, multi-step tasks")
    print("   • Transparent reasoning process")
    print("   • Reliable and verifiable results")

# =============================================================================
# BUSINESS IMPACT EXAMPLES
# =============================================================================

def show_business_applications():
    """
    Show concrete business applications of Agentic AI
    across different departments
    """
    print("\n" + "="*60)
    print("💼 AGENTIC AI IN BUSINESS")
    print("="*60)
    
    applications = {
        "Finance": [
            "Expense report analysis and anomaly detection",
            "Budget planning with market research integration",
            "Invoice processing with vendor verification"
        ],
        "HR": [
            "Resume screening with skill gap analysis",
            "Interview scheduling with candidate research",
            "Employee onboarding workflow automation"
        ],
        "Marketing": [
            "Campaign performance analysis and optimization",
            "Competitor research and strategy recommendations",
            "Content creation with SEO optimization"
        ],
        "Operations": [
            "Supply chain monitoring and proactive ordering",
            "Quality control with root cause analysis",
            "Process optimization with cost-benefit analysis"
        ]
    }
    
    for department, tasks in applications.items():
        print(f"\n🏢 {department} Department:")
        for task in tasks:
            print(f"   • {task}")
    
    print("\n🎯 Common Pattern: Research → Analyze → Decide → Act → Verify")

# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

def run_hour1_q1_demo():
    """
    Main function to run the complete Hour 1 Q1 demonstration
    """
    print("🚀 WELCOME TO AGENTIC AI MASTERY")
    print("Hour 1, Quarter 1: Understanding Agentic AI")
    print("=" * 60)
    
    # 1. Environment Setup
    if not setup_environment():
        return
    
    # 2. Show AI Evolution
    print("\n🔄 THE EVOLUTION OF AI SYSTEMS")
    demonstrate_traditional_ai()
    demonstrate_generative_ai() 
    demonstrate_agentic_ai()
    
    # 3. Explain Core Concepts
    explain_react_pattern()
    
    # 4. Business Applications
    show_business_applications()
    
    # 5. What's Next
    print("\n" + "="*60)
    print("🎉 CONGRATULATIONS!")
    print("="*60)
    print("You now understand:")
    print("✅ The evolution from Traditional → Generative → Agentic AI")
    print("✅ The ReAct pattern that powers intelligent agents")
    print("✅ Real business applications across departments")
    print("✅ Your development environment is ready")
    print()
    print("🚀 Next Up (Q2): Building your first reasoning agent!")
    print("   → You'll create an AI that thinks step-by-step")
    print("   → It will solve complex problems autonomously")
    print("   → You'll see ReAct in action with real code")

# =============================================================================
# INTERACTIVE EXPLORATION
# =============================================================================

def interactive_exploration():
    """
    Optional interactive section for curious participants
    """
    print("\n🔍 INTERACTIVE EXPLORATION (Optional)")
    print("Try asking these questions during the session:")
    
    questions = [
        "What makes an AI system 'agentic' vs just 'smart'?",
        "How does ReAct compare to how humans solve problems?",
        "What business process in our company could benefit from this?",
        "What are the risks of autonomous AI agents?",
        "How do we ensure agents make good decisions?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")

if __name__ == "__main__":
    # Run the complete Hour 1 Q1 demonstration
    run_hour1_q1_demo()
    
    # Optional interactive section
    interactive_exploration()
    
    print("\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 1 Q2: Building Your First Agent!")