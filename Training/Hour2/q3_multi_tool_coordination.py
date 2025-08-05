"""
Hour 2 - Quarter 3: Multi-Tool Coordination
===========================================

Learning Objectives:
- Implement intelligent workflow planning and tool coordination
- Learn advanced tool chaining and sequencing strategies
- Build agents that optimize tool usage for different scenarios
- Create production-ready multi-tool automation systems

Duration: 15 minutes
Technical Skills: Workflow orchestration, intelligent planning, tool optimization
"""

import os
import json
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# ENHANCED TOOL CLASSES WITH COORDINATION METADATA
# =============================================================================

class CoordinatedFileProcessor:
    """Enhanced file processor with coordination metadata"""
    
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.tool_name = "FileAnalyzer"
        self.coordination_hints = {
            "best_used_first": ["document_analysis", "report_review"],
            "pairs_well_with": ["WebSearch", "Calculator"],
            "output_feeds_into": ["research_queries", "calculation_inputs"]
        }
    
    def analyze_document(self, file_path):
        """Analyze document with coordination metadata"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {
                    "success": False,
                    "content": "",
                    "analysis": f"File not found: {file_path}",
                    "coordination_data": {
                        "suggests_next_tools": [],
                        "extracted_queries": [],
                        "numerical_data": []
                    }
                }
            
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Enhanced AI analysis with coordination hints
            analysis_prompt = f"""Analyze this business document and provide:

1. DOCUMENT ANALYSIS:
   - Document Type and Purpose
   - Key Topics and Themes
   - Important Data Points (numbers, dates, metrics)
   - Business Insights and Implications

2. COORDINATION SUGGESTIONS:
   - What web searches would enhance this analysis?
   - What calculations should be performed on the data?
   - What external information is needed?

Document Content:
{content[:4000]}{'...' if len(content) > 4000 else ''}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst AI that suggests follow-up actions for comprehensive analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            ai_analysis = response.choices[0].message.content
            
            # Extract coordination data
            coordination_data = self._extract_coordination_data(ai_analysis, content)
            
            return {
                "success": True,
                "content": content,
                "file_name": path.name,
                "analysis": ai_analysis,
                "coordination_data": coordination_data,
                "summary": f"Analyzed {path.name} with coordination suggestions"
            }
            
        except Exception as e:
            return {
                "success": False,
                "analysis": f"File analysis failed: {e}",
                "coordination_data": {"suggests_next_tools": [], "extracted_queries": [], "numerical_data": []},
                "summary": f"Analysis error: {e}"
            }
    
    def _extract_coordination_data(self, analysis, content):
        """Extract coordination suggestions from analysis"""
        # Simple extraction of numbers and potential search terms
        import re
        
        numbers = re.findall(r'\$?[\d,]+\.?\d*', content)
        
        coordination_data = {
            "suggests_next_tools": [],
            "extracted_queries": [],
            "numerical_data": numbers[:5]  # First 5 numbers found
        }
        
        # Suggest tools based on content analysis
        if "competitor" in analysis.lower():
            coordination_data["suggests_next_tools"].append("WebSearch")
            coordination_data["extracted_queries"].append("competitor news 2024")
        
        if "market" in analysis.lower():
            coordination_data["suggests_next_tools"].append("WebSearch")
            coordination_data["extracted_queries"].append("market trends 2024")
        
        if numbers:
            coordination_data["suggests_next_tools"].append("Calculator")
        
        return coordination_data

class CoordinatedWebSearch:
    """Enhanced web search with coordination awareness"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        self.tool_name = "WebSearch"
        
    def search(self, query, context=None):
        """Enhanced search with context awareness"""
        if not self.api_key:
            return {"success": False, "search_summary": "Search unavailable", "coordination_data": {}}
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"query": query, "max_results": 2}
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                summary = f"Found {len(results)} results for '{query}':\n"
                for i, result in enumerate(results, 1):
                    summary += f"{i}. {result.get('title', 'No title')[:60]}...\n"
                
                # Coordination suggestions based on search results
                coordination_data = {
                    "suggests_calculations": self._suggest_calculations(results),
                    "suggests_file_analysis": context and "document" in context.lower(),
                    "follow_up_searches": self._suggest_follow_up_searches(query, results)
                }
                
                return {
                    "success": True,
                    "search_summary": summary,
                    "coordination_data": coordination_data,
                    "results": results
                }
            else:
                return {"success": False, "search_summary": f"Search failed: {query}", "coordination_data": {}}
                
        except Exception as e:
            return {"success": False, "search_summary": f"Search error: {e}", "coordination_data": {}}
    
    def _suggest_calculations(self, results):
        """Suggest calculations based on search results"""
        calc_suggestions = []
        for result in results:
            content = result.get("content", "").lower()
            if "%" in content or "growth" in content:
                calc_suggestions.append("growth_percentage")
            if "$" in content or "revenue" in content:
                calc_suggestions.append("financial_comparison")
        return calc_suggestions
    
    def _suggest_follow_up_searches(self, original_query, results):
        """Suggest follow-up searches"""
        suggestions = []
        if "market" in original_query.lower():
            suggestions.append(f"{original_query} forecast")
        if "competitor" in original_query.lower():
            suggestions.append(f"{original_query} strategy")
        return suggestions[:2]

class CoordinatedCalculator:
    """Enhanced calculator with coordination awareness"""
    
    def __init__(self):
        self.tool_name = "Calculator"
    
    def calculate(self, expression, context=None):
        """Enhanced calculation with context awareness"""
        try:
            # Clean up expression
            clean_expr = expression.strip()
            open_parens = clean_expr.count('(')
            close_parens = clean_expr.count(')')
            if open_parens > close_parens:
                clean_expr += ')' * (open_parens - close_parens)
            
            allowed_names = {
                "__builtins__": None,
                "abs": abs, "round": round, "pow": pow,
                "max": max, "min": min
            }
            
            result = eval(clean_expr, allowed_names, {})
            
            # Coordination suggestions
            coordination_data = {
                "result_type": self._classify_result(result, expression),
                "suggests_research": "%" in expression and result > 10,
                "suggests_comparison": context and "market" in context.lower()
            }
            
            return {
                "success": True,
                "result": result,
                "formatted": f"{expression} = {result}",
                "coordination_data": coordination_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "formatted": f"Calculator Error: {e}",
                "coordination_data": {}
            }
    
    def _classify_result(self, result, expression):
        """Classify the type of calculation result"""
        if "%" in expression or (isinstance(result, (int, float)) and 0 < result < 100):
            return "percentage"
        elif "$" in expression or result > 1000:
            return "financial"
        else:
            return "general"

# =============================================================================
# WORKFLOW COORDINATOR
# =============================================================================

class WorkflowCoordinator:
    """
    Intelligent workflow coordinator that plans and optimizes tool usage
    """
    
    def __init__(self):
        self.file_processor = CoordinatedFileProcessor()
        self.web_search = CoordinatedWebSearch()
        self.calculator = CoordinatedCalculator()
        
        # Workflow patterns based on business scenarios
        self.workflow_patterns = {
            "financial_analysis": ["FileAnalyzer", "Calculator", "WebSearch"],
            "competitive_intelligence": ["FileAnalyzer", "WebSearch", "Calculator"],
            "market_research": ["WebSearch", "Calculator", "FileAnalyzer"],
            "document_review": ["FileAnalyzer", "WebSearch"],
            "data_analysis": ["FileAnalyzer", "Calculator", "WebSearch"]
        }
    
    def suggest_workflow(self, user_question):
        """
        Analyze user question and suggest optimal workflow
        """
        question_lower = user_question.lower()
        
        # Pattern matching for workflow suggestions
        if any(word in question_lower for word in ["financial", "budget", "revenue", "profit"]):
            return {
                "pattern": "financial_analysis",
                "suggested_sequence": self.workflow_patterns["financial_analysis"],
                "reasoning": "Financial questions benefit from document analysis, calculations, and market research"
            }
        elif any(word in question_lower for word in ["competitor", "competition", "market position"]):
            return {
                "pattern": "competitive_intelligence", 
                "suggested_sequence": self.workflow_patterns["competitive_intelligence"],
                "reasoning": "Competitive analysis needs internal data, external research, and comparison calculations"
            }
        elif any(word in question_lower for word in ["market", "industry", "trend"]):
            return {
                "pattern": "market_research",
                "suggested_sequence": self.workflow_patterns["market_research"],
                "reasoning": "Market questions start with external research, then analysis and documentation"
            }
        elif any(word in question_lower for word in ["analyze", "review", "document"]):
            return {
                "pattern": "document_review",
                "suggested_sequence": self.workflow_patterns["document_review"],
                "reasoning": "Document-focused questions need file analysis and supporting research"
            }
        else:
            return {
                "pattern": "data_analysis",
                "suggested_sequence": self.workflow_patterns["data_analysis"],
                "reasoning": "General analysis benefits from comprehensive multi-tool approach"
            }
    
    def optimize_tool_sequence(self, initial_sequence, previous_results):
        """
        Optimize tool sequence based on previous results and coordination data
        """
        optimized_sequence = initial_sequence.copy()
        
        # Check if previous tool results suggest reordering
        for result in previous_results:
            if hasattr(result, 'coordination_data'):
                coord_data = result.coordination_data
                if coord_data.get("suggests_next_tools"):
                    # Move suggested tools up in priority
                    for suggested_tool in coord_data["suggests_next_tools"]:
                        if suggested_tool in optimized_sequence:
                            optimized_sequence.remove(suggested_tool)
                            optimized_sequence.insert(0, suggested_tool)
        
        return optimized_sequence

# =============================================================================
# COORDINATED MULTI-TOOL AGENT
# =============================================================================

class CoordinatedAgent:
    """
    Advanced agent with intelligent multi-tool coordination
    Plans workflows, optimizes tool sequences, and learns from results
    """
    
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.coordinator = WorkflowCoordinator()
        
        self.system_prompt = """You are an intelligent business analyst agent with advanced workflow coordination capabilities.

AVAILABLE TOOLS WITH COORDINATION:
1. FileAnalyzer(file_path) - Analyzes documents and suggests follow-up actions
2. WebSearch(query) - Searches internet and suggests related research  
3. Calculator(expression) - Performs calculations and suggests comparisons

WORKFLOW INTELLIGENCE:
- Plan optimal tool sequences based on question type
- Use coordination data from each tool to guide next steps
- Adapt workflow based on intermediate results
- Combine insights from multiple tools for comprehensive analysis

COORDINATION PRINCIPLES:
1. Start with the most informative tool for the question type
2. Use each tool's coordination suggestions to guide next steps
3. Chain tools logically: Document → Research → Calculate → Verify
4. Synthesize results from all tools for final answer

FORMAT YOUR RESPONSE:
Thought: [Analysis of question and workflow planning]
Workflow Plan: [Planned sequence of tools and reasoning]
Action: [First tool usage]
Observation: [Result and coordination suggestions]
[Continue with coordinated tool usage]

Final Answer: [Comprehensive synthesis of all tool results]

WORKFLOW EXAMPLES:
Financial Analysis: FileAnalyzer(report) → Calculator(metrics) → WebSearch(benchmarks)
Competitive Intel: FileAnalyzer(analysis) → WebSearch(recent news) → Calculator(comparisons)
Market Research: WebSearch(trends) → Calculator(projections) → FileAnalyzer(strategy doc)
"""
    
    def coordinated_analysis(self, user_question, max_steps=15):
        """
        Perform coordinated multi-tool analysis with intelligent workflow planning
        """
        print(f"\n🤖 Coordinated Agent received: {user_question}")
        print("🧠 Planning optimal workflow strategy...\n")
        
        # Step 1: Suggest initial workflow
        workflow_suggestion = self.coordinator.suggest_workflow(user_question)
        print(f"📋 Suggested Workflow Pattern: {workflow_suggestion['pattern']}")
        print(f"🔄 Planned Sequence: {' → '.join(workflow_suggestion['suggested_sequence'])}")
        print(f"💡 Reasoning: {workflow_suggestion['reasoning']}\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Business Analysis Request: {user_question}\n\nSuggested Workflow: {workflow_suggestion['suggested_sequence']}\nReasoning: {workflow_suggestion['reasoning']}"}
        ]
        
        step_count = 0
        coordination_results = []
        tools_used = {"file": 0, "search": 0, "calc": 0}
        
        while step_count < max_steps:
            step_count += 1
            print(f"🔄 Coordinated Step {step_count}:")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=700
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                
                # Process coordinated tool calls
                tool_result = None
                coordination_data = {}
                
                # Enhanced tool processing with coordination
                if "FileAnalyzer(" in agent_response:
                    tools_used["file"] += 1
                    file_path = self._extract_file_path(agent_response)
                    if file_path:
                        result = self.coordinator.file_processor.analyze_document(file_path)
                        tool_result = result["summary"]
                        coordination_data = result.get("coordination_data", {})
                        coordination_results.append(result)
                        print(f"\n📄 File Analysis: {tool_result}")
                        if coordination_data.get("suggests_next_tools"):
                            print(f"🔗 Coordination Suggestion: Use {coordination_data['suggests_next_tools']} next")
                
                elif "WebSearch(" in agent_response:
                    tools_used["search"] += 1
                    query = self._extract_search_query(agent_response)
                    if query:
                        context = " ".join([r.get("summary", "") for r in coordination_results[-2:]])
                        result = self.coordinator.web_search.search(query, context)
                        tool_result = result["search_summary"]
                        coordination_data = result.get("coordination_data", {})
                        print(f"\n🌐 Web Search: {tool_result}")
                        if coordination_data.get("suggests_calculations"):
                            print(f"🔗 Coordination Suggestion: Calculate {coordination_data['suggests_calculations']}")
                
                elif "Calculator(" in agent_response:
                    tools_used["calc"] += 1
                    expression = self._extract_calculator_expression(agent_response)
                    if expression:
                        context = " ".join([r.get("summary", "") for r in coordination_results[-2:]])
                        result = self.coordinator.calculator.calculate(expression, context)
                        tool_result = result["formatted"]
                        coordination_data = result.get("coordination_data", {})
                        print(f"\n🔢 Calculator: {tool_result}")
                        if coordination_data.get("suggests_research"):
                            print(f"🔗 Coordination Suggestion: Research market benchmarks")
                
                # Update conversation with coordination context
                messages.append({"role": "assistant", "content": agent_response})
                if tool_result:
                    coord_info = f"Tool Result: {tool_result}"
                    if coordination_data:
                        coord_info += f"\nCoordination Data: {coordination_data}"
                    messages.append({"role": "user", "content": coord_info})
                
                print("-" * 50)
                
                # Check for completion
                if "Final Answer:" in agent_response:
                    total_tools = sum(tools_used.values())
                    print(f"✅ Coordinated analysis complete! Used {total_tools} tools with intelligent coordination")
                    print(f"📊 Tool Usage: {tools_used['file']} file, {tools_used['search']} search, {tools_used['calc']} calc")
                    return self._extract_final_answer(agent_response)
                
                # Continue coordination if no final answer
                if not tool_result:
                    messages.append({
                        "role": "user",
                        "content": "Continue coordinated analysis or provide Final Answer if complete."
                    })
                
            except Exception as e:
                print(f"❌ Error in coordinated step {step_count}: {e}")
                return f"Coordination error: {e}"
        
        print("⚠️ Reached maximum coordination steps")
        return "Coordinated analysis incomplete - reached step limit"
    
    def _extract_file_path(self, response):
        """Extract file path from response"""
        import re
        match = re.search(r'FileAnalyzer\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
    def _extract_search_query(self, response):
        """Extract search query from response"""
        import re
        match = re.search(r'WebSearch\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
    def _extract_calculator_expression(self, response):
        """Extract calculator expression from response"""
        import re
        match = re.search(r'Calculator\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
    def _extract_final_answer(self, response):
        """Extract final answer from response"""
        lines = response.split('\n')
        for line in lines:
            if line.startswith("Final Answer:"):
                return line.replace("Final Answer:", "").strip()
        return "Final answer format not found"

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_coordination_need():
    """Show why intelligent coordination matters"""
    print("🔗 WHY INTELLIGENT COORDINATION MATTERS")
    print("=" * 60)
    
    scenarios = [
        {
            "scenario": "Uncoordinated Approach",
            "workflow": "Random tool usage → Missed connections → Incomplete analysis",
            "result": "Agent uses tools individually, misses insights",
            "efficiency": "Low - tools used suboptimally"
        },
        {
            "scenario": "Basic Coordination",
            "workflow": "Fixed sequence → Some connections → Better analysis", 
            "result": "Agent follows preset patterns, decent results",
            "efficiency": "Medium - predictable but not adaptive"
        },
        {
            "scenario": "Intelligent Coordination",
            "workflow": "Dynamic planning → Tool suggestions → Optimized analysis",
            "result": "Agent adapts workflow based on results, comprehensive insights",
            "efficiency": "High - optimal tool usage and sequencing"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Level {i}: {scenario['scenario']}")
        print(f"   Workflow: {scenario['workflow']}")
        print(f"   Result: {scenario['result']}")
        print(f"   Efficiency: {scenario['efficiency']}")
    
    print("\n🎯 Intelligent coordination maximizes tool synergy and analysis quality!")

def demonstrate_workflow_patterns():
    """Show different workflow patterns for different business scenarios"""
    print("\n🔄 INTELLIGENT WORKFLOW PATTERNS")
    print("=" * 60)
    
    patterns = [
        {
            "scenario": "Financial Analysis",
            "optimal_flow": "Document Analysis → Calculate Metrics → Research Benchmarks",
            "why_optimal": "Internal data first, then calculations, then external comparison",
            "coordination_benefits": "Document suggests calculations, results guide research"
        },
        {
            "scenario": "Competitive Intelligence", 
            "optimal_flow": "Analyze Internal Data → Research Competitor Updates → Calculate Gaps",
            "why_optimal": "Know our position, find competitor changes, quantify differences",
            "coordination_benefits": "Internal analysis guides research focus, results enable calculations"
        },
        {
            "scenario": "Market Research",
            "optimal_flow": "External Research → Analyze Projections → Review Strategy Docs",
            "why_optimal": "Market context first, then implications, then strategic alignment",
            "coordination_benefits": "Market data guides calculations, results inform document analysis"
        }
    ]
    
    for pattern in patterns:
        print(f"\n🎯 {pattern['scenario']}:")
        print(f"   Optimal Flow: {pattern['optimal_flow']}")
        print(f"   Why Optimal: {pattern['why_optimal']}")
        print(f"   Coordination Benefits: {pattern['coordination_benefits']}")
    
    print("\n🔗 Each workflow optimizes tool synergy for maximum business insight!")

# =============================================================================
# TESTING COORDINATION
# =============================================================================

def test_coordination_capabilities():
    """Test coordinated agent with complex business scenarios"""
    print("\n🧪 TESTING COORDINATED MULTI-TOOL CAPABILITIES")
    print("=" * 70)
    
    # Ensure sample documents exist
    from q2_file_processing_integration import create_sample_documents
    create_sample_documents()
    
    agent = CoordinatedAgent()
    
    test_cases = [
        {
            "name": "Advanced Financial Analysis",
            "question": "Analyze our quarterly performance in 'sample_documents/quarterly_report.txt', research current software industry benchmarks, and calculate where we stand competitively. What strategic actions should we take?",
            "expected_coordination": "File → Calculation → Research → Strategic synthesis"
        },
        {
            "name": "Comprehensive Competitive Intelligence",
            "question": "Review our competitor analysis in 'sample_documents/competitor_analysis.txt', find the latest news about these companies, and calculate market share implications. How should we respond?",
            "expected_coordination": "File → Research → Calculation → Strategic recommendations"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Coordination Test {i}: {test['name']}")
        print(f"🔗 Expected Coordination: {test['expected_coordination']}")
        print(f"❓ Complex Question: {test['question'][:100]}...")
        
        result = agent.coordinated_analysis(test['question'])
        print(f"🏆 Coordinated Result: {result[:200]}...")
        print("\n" + "=" * 80)
        
        if i < len(test_cases):
            input("Press Enter to continue to next coordination test...")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def coordination_workshop():
    """Interactive workshop with coordinated multi-tool capabilities"""
    print("\n🎯 MULTI-TOOL COORDINATION WORKSHOP")
    print("=" * 60)
    
    agent = CoordinatedAgent()
    
    print("Test your coordinated agent with complex multi-step business scenarios!")
    print("Coordination Excellence Challenges:")
    print("• Complex financial analysis requiring all three tools")
    print("• Competitive intelligence with document + research + calculations")
    print("• Strategic planning scenarios requiring comprehensive analysis")
    print("• Multi-faceted business questions needing optimal tool sequencing")
    print("\nType 'exit' to finish this quarter.")
    
    while True:
        user_question = input("\n💬 Your complex coordination challenge: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Exceptional! Your agent now coordinates tools with business intelligence!")
            break
        
        if user_question.strip():
            result = agent.coordinated_analysis(user_question)
            print(f"\n🎯 Coordinated Analysis Result: {result}")
        else:
            print("Please enter a complex question requiring multiple coordinated tools.")

# =============================================================================
# MAIN WORKSHOP FUNCTION  
# =============================================================================

def run_hour2_q3_workshop():
    """Main function for Hour 2 Q3 workshop"""
    print("🚀 HOUR 2 - QUARTER 3: MULTI-TOOL COORDINATION")
    print("=" * 70)
    
    # Step 1: Show coordination importance
    demonstrate_coordination_need()
    
    # Step 2: Show workflow patterns
    demonstrate_workflow_patterns()
    
    # Step 3: Test coordination capabilities
    test_coordination_capabilities()
    
    # Step 4: Interactive workshop
    coordination_workshop()
    
    # Step 5: Quarter completion and Q4 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 3 COMPLETE!")
    print("=" * 60)
    print("Multi-Tool Coordination Achievements:")
    print("✅ Intelligent workflow planning and optimization")
    print("✅ Advanced tool chaining and sequencing")
    print("✅ Coordination metadata and suggestion systems")
    print("✅ Adaptive workflow based on intermediate results")
    print("✅ Production-ready multi-tool automation")
    
    print("\n🏆 Your Agent Now Demonstrates:")
    print("   → Intelligent workflow planning for different business scenarios")
    print("   → Optimal tool sequencing and coordination")
    print("   → Adaptive analysis based on intermediate results")
    print("   → Comprehensive business intelligence synthesis")
    print("   → Production-ready automation for complex tasks")
    
    print("\n📈 Complete Multi-Tool Evolution:")
    print("   Hour 1: Individual Tools → Reasoning + Calculator")
    print("   Hour 2 Q1: + Web Search → Internet Research")
    print("   Hour 2 Q2: + File Processing → Document Intelligence")
    print("   Hour 2 Q3: + Coordination → Intelligent Workflows")
    print("   Hour 2 Q4: Advanced Reasoning → Business Intelligence")
    
    print("\n🚀 Coming Up in Q4: Advanced Reasoning & Workflows")
    print("   → Complex business scenario automation")
    print("   → Advanced reasoning and decision-making")
    print("   → End-to-end business process management")
    print("   → Enterprise-ready intelligent agent systems")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 2 Q4: Advanced Reasoning & Business Intelligence!")

if __name__ == "__main__":
    # Run the complete Hour 2 Q3 workshop
    run_hour2_q3_workshop()