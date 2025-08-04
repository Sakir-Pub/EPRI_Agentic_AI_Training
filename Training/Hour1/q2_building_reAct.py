"""
Hour 1 - Quarter 2: Building Your First Reasoning Agent
======================================================

Learning Objectives:
- Implement the ReAct pattern in actual code
- Build an AI agent that thinks step-by-step
- Understand prompt engineering for reasoning
- Create your first autonomous problem-solving AI

Duration: 15 minutes
Technical Skills: OpenAI API, ReAct implementation, prompt engineering
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# BASIC REASONING AGENT CLASS
# =============================================================================

class BasicReasoningAgent:
    """
    Your first Agentic AI! This agent implements the ReAct pattern:
    - REASON: Think about the problem step by step
    - ACT: Take a specific action or calculation
    - OBSERVE: Analyze the result and decide next steps
    """
    
    def __init__(self):
        """Initialize the agent with OpenAI connection"""
        load_dotenv()
        self.client = OpenAI()
        self.conversation_history = []
        
        # ReAct Prompt Template - This is the "brain" of our agent
        self.system_prompt = """You are a helpful reasoning agent that thinks step-by-step.

IMPORTANT: You must follow the ReAct pattern for EVERY response:

1. REASON (Thought): Think about what you need to do
2. ACT (Action): Describe the specific action you're taking  
3. OBSERVE (Observation): Analyze the result
4. Repeat until you reach a Final Answer

Format your response EXACTLY like this:

Thought: [Your reasoning about what to do next]
Action: [The specific action you're taking]
Observation: [What you learned from the action]

Continue this pattern until you can give a Final Answer.

Example:
Thought: I need to calculate 20% of $150 for a tip calculation.
Action: Calculate 150 × 0.20
Observation: The result is $30.00

Thought: Let me verify this makes sense - $30 on $150 is indeed 20%.
Action: Verify: 30 ÷ 150 = 0.20
Observation: Confirmed, this is exactly 20%.

Final Answer: A 20% tip on $150 is $30.00.

Always end with "Final Answer: [your complete answer]"
"""
    
    def think_step_by_step(self, user_question, max_steps=5):
        """
        This is where the magic happens! The agent reasons through problems
        using the ReAct pattern, just like we saw in the demo.
        """
        print(f"\n🤖 Agent received question: {user_question}")
        print("🧠 Let me think through this step by step...\n")
        
        # Prepare the conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]
        
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"🔄 Step {step_count}:")
            
            try:
                # Get AI response
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent reasoning
                    max_tokens=300
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                print("-" * 50)
                
                # Add to conversation history
                messages.append({"role": "assistant", "content": agent_response})
                
                # Check if we reached a final answer
                if "Final Answer:" in agent_response:
                    print("✅ Agent reached final answer!")
                    return self.extract_final_answer(agent_response)
                
                # If no final answer, continue the reasoning
                messages.append({
                    "role": "user", 
                    "content": "Continue reasoning if needed, or provide Final Answer."
                })
                
                time.sleep(1)  # Brief pause for readability
                
            except Exception as e:
                print(f"❌ Error in reasoning step: {e}")
                return "Error occurred during reasoning"
        
        print("⚠️ Agent reached maximum reasoning steps")
        return "Could not complete reasoning within step limit"
    
    def extract_final_answer(self, response):
        """Extract the final answer from the agent's response"""
        lines = response.split('\n')
        for line in lines:
            if line.startswith("Final Answer:"):
                return line.replace("Final Answer:", "").strip()
        return "Final answer not clearly formatted"

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_agent_creation():
    """Show how we build our reasoning agent"""
    print("🔨 BUILDING YOUR FIRST REASONING AGENT")
    print("=" * 60)
    
    print("Step 1: Creating the agent class...")
    print("✅ ReAct pattern implemented")
    print("✅ OpenAI API connection established")
    print("✅ Step-by-step reasoning logic ready")
    
    print("\nStep 2: Setting up the 'brain' (system prompt)...")
    print("✅ Instructions for ReAct pattern")
    print("✅ Format requirements defined")
    print("✅ Examples provided")
    
    print("\nStep 3: Creating reasoning loop...")
    print("✅ Multi-step thinking capability")
    print("✅ Error handling included")
    print("✅ Final answer extraction")
    
    print("\n🎉 Your reasoning agent is ready!")

def run_agent_tests():
    """Test our agent with progressively harder problems"""
    print("\n🧪 TESTING YOUR REASONING AGENT")
    print("=" * 60)
    
    # Create our agent
    agent = BasicReasoningAgent()
    
    # Test cases - from simple to complex
    test_cases = [
        {
            "question": "If I save $500 per month for 6 months, how much will I have saved?",
            "difficulty": "Simple",
            "learning_point": "Basic multiplication and reasoning"
        },
        {
            "question": "I have $5,000 in savings. If I spend $1,200 on a laptop and then save $300 more, what's my new balance?",
            "difficulty": "Medium", 
            "learning_point": "Multi-step calculations with verification"
        },
        {
            "question": "Our team budget is $50,000. We spend 40% on salaries, 25% on equipment, and 15% on training. How much is left for other expenses?",
            "difficulty": "Complex",
            "learning_point": "Percentage calculations and remainder logic"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['difficulty']} Problem")
        print(f"🎯 Learning Point: {test_case['learning_point']}")
        print(f"❓ Question: {test_case['question']}")
        
        # Let the agent solve it
        final_answer = agent.think_step_by_step(test_case['question'])
        
        print(f"🏆 Final Result: {final_answer}")
        print("\n" + "=" * 80)
        
        # Pause between tests for better readability
        if i < len(test_cases):
            input("Press Enter to continue to next test...")

# =============================================================================
# INTERACTIVE WORKSHOP FUNCTIONS
# =============================================================================

def workshop_challenge():
    """Interactive challenge for workshop participants"""
    print("\n🎯 WORKSHOP CHALLENGE - YOUR TURN!")
    print("=" * 60)
    
    agent = BasicReasoningAgent()
    
    print("Now it's your turn to test the agent you just built!")
    print("Try asking business-related questions that require step-by-step thinking.")
    print("\nSuggested challenges:")
    print("• Budget calculations and planning")
    print("• ROI and profit/loss scenarios") 
    print("• Resource allocation problems")
    print("• Time and cost estimation")
    print("\nType 'exit' to finish the workshop section.")
    
    while True:
        user_question = input("\n💬 Your question for the agent: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Great work! You've successfully built and tested your first reasoning agent!")
            break
        
        if user_question.strip():
            final_answer = agent.think_step_by_step(user_question)
            print(f"\n🎯 Agent's Final Answer: {final_answer}")
        else:
            print("Please enter a question for the agent to solve.")

def explain_what_we_built():
    """Explain the technical concepts we just implemented"""
    print("\n🔬 WHAT YOU JUST BUILT - TECHNICAL BREAKDOWN")
    print("=" * 60)
    
    concepts = {
        "ReAct Pattern Implementation": [
            "System prompt that enforces step-by-step thinking",
            "Conversation loop that maintains reasoning chain", 
            "Response parsing to identify completion"
        ],
        "Prompt Engineering": [
            "Structured instructions for AI behavior",
            "Format requirements and examples",
            "Temperature control for consistent reasoning"
        ],
        "Agent Architecture": [
            "Stateful conversation management",
            "Error handling and recovery",
            "Modular design for easy extension"
        ],
        "Business Applications": [
            "Multi-step problem solving",
            "Verification and quality assurance",
            "Transparent reasoning process"
        ]
    }
    
    for concept, details in concepts.items():
        print(f"\n🔧 {concept}:")
        for detail in details:
            print(f"   ✅ {detail}")
    
    print(f"\n🎓 Key Skills Gained:")
    print("   • Built your first autonomous reasoning system")
    print("   • Implemented ReAct pattern in production code")
    print("   • Understanding of prompt engineering principles")
    print("   • Foundation for building more complex agents")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour1_q2_workshop():
    """Main function for Hour 1 Q2 workshop"""
    print("🚀 HOUR 1 - QUARTER 2: BUILD YOUR FIRST REASONING AGENT")
    print("=" * 70)
    
    # Step 1: Show what we're building
    demonstrate_agent_creation()
    
    # Step 2: Build and test the agent
    run_agent_tests()
    
    # Step 3: Interactive challenge
    workshop_challenge()
    
    # Step 4: Explain what we accomplished
    explain_what_we_built()
    
    # Step 5: Setup for next quarter
    print("\n" + "=" * 60)
    print("🎉 QUARTER 2 COMPLETE!")
    print("=" * 60)
    print("Your Achievements:")
    print("✅ Built your first reasoning agent from scratch")
    print("✅ Implemented the ReAct pattern in code")
    print("✅ Tested with real business problems")
    print("✅ Understanding of prompt engineering")
    print("✅ Foundation for advanced agent capabilities")
    
    print("\n🚀 Coming Up in Q3: Enhanced Reasoning & Error Handling")
    print("   → Make your agent more robust and reliable")
    print("   → Add self-correction capabilities")
    print("   → Handle edge cases and errors gracefully")
    print("   → Build production-ready agents")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 1 Q3!")

# =============================================================================
# QUICK DEMO FUNCTION (for instructor)
# =============================================================================

def quick_demo():
    """Quick demonstration function for instructor use"""
    print("🎬 QUICK DEMO: ReAct Agent in Action")
    print("=" * 50)
    
    agent = BasicReasoningAgent()
    demo_question = "If our company grows revenue by 25% each quarter for 4 quarters starting from $100,000, what will our revenue be at the end?"
    
    print(f"Demo Question: {demo_question}")
    result = agent.think_step_by_step(demo_question)
    print(f"Final Result: {result}")

if __name__ == "__main__":
    # Run the complete Hour 1 Q2 workshop
    run_hour1_q2_workshop()
    
    # Uncomment for quick demo version:
    # quick_demo()