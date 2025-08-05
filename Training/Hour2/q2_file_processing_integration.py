"""
Hour 2 - Quarter 2: File Processing Integration
===============================================

Learning Objectives:
- Add document analysis and processing capabilities
- Learn intelligent file summarization techniques
- Integrate file processing with web search and calculations
- Build agents that can analyze business documents

Duration: 15 minutes
Technical Skills: File I/O, document processing, content analysis, multi-tool coordination
"""

import os
import json
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# =============================================================================
# FILE PROCESSING TOOLS
# =============================================================================

class FileProcessor:
    """
    Professional file processing tool for business documents
    Supports TXT, PDF, DOCX, and other common formats
    """
    
    def __init__(self):
        """Initialize the file processor"""
        load_dotenv()
        self.client = OpenAI()
        self.supported_formats = ['.txt', '.md', '.csv', '.json', '.py', '.js', '.html']
        
    def read_file(self, file_path):
        """
        Read and process various file formats
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "content": "",
                    "error": f"File not found: {file_path}",
                    "summary": f"Could not locate file: {file_path}"
                }
            
            if path.suffix.lower() not in self.supported_formats:
                return {
                    "success": False,
                    "content": "",
                    "error": f"Unsupported file format: {path.suffix}",
                    "summary": f"Cannot process {path.suffix} files yet"
                }
            
            # Read file content
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if not content.strip():
                return {
                    "success": False,
                    "content": "",
                    "error": "File is empty",
                    "summary": f"File {path.name} contains no content"
                }
            
            # Create basic summary
            summary = self._create_file_summary(path.name, content)
            
            return {
                "success": True,
                "content": content,
                "file_name": path.name,
                "file_size": len(content),
                "summary": summary,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e),
                "summary": f"Error reading file: {e}"
            }
    
    def analyze_document(self, file_path):
        """
        Analyze document content using AI for intelligent insights
        """
        file_result = self.read_file(file_path)
        
        if not file_result["success"]:
            return file_result
        
        try:
            # Use AI to analyze the document
            analysis_prompt = f"""Analyze this business document and provide:

1. Document Type: What kind of document is this?
2. Key Topics: Main subjects covered
3. Important Data: Numbers, dates, metrics mentioned
4. Business Insights: What are the key takeaways?
5. Action Items: Any tasks or decisions mentioned

Document Content:
{file_result['content'][:3000]}{'...' if len(file_result['content']) > 3000 else ''}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business document analyst. Provide concise, structured analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            ai_analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "content": file_result["content"],
                "file_name": file_result["file_name"],
                "analysis": ai_analysis,
                "summary": f"Analyzed {file_result['file_name']}: {ai_analysis[:100]}...",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": file_result["content"],
                "analysis": f"AI analysis failed: {e}",
                "summary": f"Basic file read successful, AI analysis failed: {e}",
                "error": str(e)
            }
    
    def _create_file_summary(self, filename, content):
        """Create a basic summary of file content"""
        lines = content.split('\n')
        word_count = len(content.split())
        
        summary = f"File: {filename}\n"
        summary += f"Lines: {len(lines)}, Words: {word_count}\n"
        summary += f"Preview: {content[:200]}..."
        
        return summary

# =============================================================================
# WEB SEARCH TOOL (from Q1)
# =============================================================================

class WebSearchTool:
    """Web search tool from Hour 2 Q1"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
    
    def search(self, query, max_results=2):
        """Perform web search - simplified for file integration focus"""
        if not self.api_key:
            return {
                "success": False,
                "search_summary": "Search unavailable - API key missing"
            }
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"query": query, "max_results": max_results}
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                summary = f"Found {len(results)} results for '{query}':\n"
                for i, result in enumerate(results, 1):
                    summary += f"{i}. {result.get('title', 'No title')[:60]}...\n"
                
                return {"success": True, "search_summary": summary}
            else:
                return {"success": False, "search_summary": f"Search failed for: {query}"}
                
        except Exception as e:
            return {"success": False, "search_summary": f"Search error: {e}"}

# =============================================================================
# CALCULATOR TOOL (from Hour 1)
# =============================================================================

def calculator_tool(expression):
    """Safe calculator tool - fixed version"""
    try:
        # Clean up common expression issues
        clean_expr = expression.strip()
        # Fix unmatched parentheses issues
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
# MULTI-TOOL AGENT WITH FILE PROCESSING
# =============================================================================

class MultiToolAgent:
    """
    Advanced agent with file processing, web search, and calculator capabilities
    Can analyze documents and combine insights with external research
    """
    
    def __init__(self):
        """Initialize the multi-tool agent"""
        load_dotenv()
        self.client = OpenAI()
        self.file_processor = FileProcessor()
        self.search_tool = WebSearchTool()
        
        # Enhanced prompt with file processing awareness
        self.system_prompt = """You are an intelligent business analyst agent with three powerful tools:

AVAILABLE TOOLS:
1. FileAnalyzer(file_path) - Analyze business documents, reports, contracts
2. WebSearch(query) - Search the internet for current information  
3. Calculator(expression) - Perform precise mathematical calculations

INTELLIGENT TOOL USAGE:
- Use FileAnalyzer for company documents, reports, contracts, data files
- Use WebSearch for current market data, competitor info, industry trends
- Use Calculator for precise mathematical analysis of data
- COMBINE tools for comprehensive analysis (e.g., analyze internal report + research market trends + calculate projections)

FORMAT YOUR RESPONSE:
Thought: [Your reasoning about what analysis is needed]
Action: [Tool usage: FileAnalyzer(path) OR WebSearch(query) OR Calculator(expression)]
Observation: [The result from the tool]
[Continue building analysis with multiple tools]

Final Answer: [Comprehensive answer combining insights from all tools used]

BUSINESS ANALYSIS PATTERNS:
- Document + Market Research: Analyze internal data, then research external trends
- Financial Analysis: Process financial documents + calculate metrics + research benchmarks
- Competitive Analysis: Review internal strategy + research competitor data + calculate comparisons

Example Multi-Tool Workflow:
Thought: I need to analyze our quarterly report and compare with market trends.
Action: FileAnalyzer(quarterly_report.txt)
Observation: [Document analysis results]

Thought: Now I need current market data for comparison.
Action: WebSearch(software industry Q3 2024 growth trends)
Observation: [Market research results]

Thought: Let me calculate growth percentages for comparison.
Action: Calculator((new_revenue - old_revenue) / old_revenue * 100)
Observation: [Calculation results]

Final Answer: Based on our internal report analysis, market research, and calculations...
"""
    
    def comprehensive_analysis(self, user_question, max_steps=12):
        """
        Perform comprehensive analysis using all available tools
        """
        print(f"\n🤖 Multi-Tool Agent received: {user_question}")
        print("🔧 Analyzing what tools are needed for comprehensive solution...\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Business Analysis Request: {user_question}"}
        ]
        
        step_count = 0
        tools_used = {"file": 0, "search": 0, "calc": 0}
        
        while step_count < max_steps:
            step_count += 1
            print(f"🔄 Step {step_count}:")
            
            try:
                # Get agent response
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600
                )
                
                agent_response = response.choices[0].message.content
                print(agent_response)
                
                # Process tool calls
                tool_result = None
                
                # Check for file analysis
                if "FileAnalyzer(" in agent_response:
                    tools_used["file"] += 1
                    file_path = self._extract_file_path(agent_response)
                    if file_path:
                        file_result = self.file_processor.analyze_document(file_path)
                        tool_result = file_result["summary"]
                        print(f"\n📄 File Analysis Result: {tool_result}")
                
                # Check for web search
                elif "WebSearch(" in agent_response:
                    tools_used["search"] += 1
                    query = self._extract_search_query(agent_response)
                    if query:
                        search_result = self.search_tool.search(query)
                        tool_result = search_result["search_summary"]
                        print(f"\n🌐 Search Result: {tool_result}")
                
                # Check for calculator
                elif "Calculator(" in agent_response:
                    tools_used["calc"] += 1
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
                    total_tools = sum(tools_used.values())
                    print(f"✅ Analysis complete! Used {total_tools} tools: {tools_used['file']} file, {tools_used['search']} search, {tools_used['calc']} calc")
                    return self._extract_final_answer(agent_response)
                
                # Continue if no final answer
                if not tool_result:
                    messages.append({
                        "role": "user",
                        "content": "Continue analysis or provide Final Answer if complete."
                    })
                
            except Exception as e:
                print(f"❌ Error in step {step_count}: {e}")
                return f"Analysis error: {e}"
        
        print("⚠️ Reached maximum analysis steps")
        return "Analysis incomplete - reached step limit"
    
    def _extract_file_path(self, response):
        """Extract file path from agent response"""
        import re
        match = re.search(r'FileAnalyzer\((.*?)\)', response)
        return match.group(1).strip('"\'') if match else None
    
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
# SAMPLE DOCUMENT CREATOR
# =============================================================================

def create_sample_documents():
    """Create sample business documents for demonstration"""
    sample_docs = {
        "quarterly_report.txt": """Q3 2024 Financial Report
Company: TechSolutions Inc.

REVENUE SUMMARY:
Q3 2024 Revenue: $2,847,500
Q2 2024 Revenue: $2,650,000
Growth: 7.45%

KEY METRICS:
- Customer Acquisition: 245 new customers
- Monthly Recurring Revenue: $950,000
- Customer Churn Rate: 3.2%
- Average Deal Size: $11,620

EXPENSES:
- Salaries: $1,125,000 (39.5%)
- Marketing: $480,000 (16.9%)
- Operations: $310,000 (10.9%)
- R&D: $285,000 (10.0%)

MARKET POSITION:
- Market share increased to 12.3%
- Main competitors: SoftwareCorp (18.5%), DataSystems (15.2%)
- Pricing 15% below market average

CHALLENGES:
- Increased competition in enterprise segment
- Rising customer acquisition costs
- Need for product differentiation

OPPORTUNITIES:
- Growing demand for AI integration
- Expansion into European market
- Strategic partnership potential
""",
        
        "competitor_analysis.txt": """Competitor Analysis Report
Date: October 2024

PRIMARY COMPETITORS:

1. SoftwareCorp
   - Market Share: 18.5%
   - Pricing: 20% premium to market
   - Strengths: Brand recognition, enterprise features
   - Weaknesses: Slow innovation, complex interface
   - Recent News: Launched AI assistant feature

2. DataSystems  
   - Market Share: 15.2%
   - Pricing: Market average
   - Strengths: Technical capabilities, integrations
   - Weaknesses: Limited marketing, poor UX
   - Recent News: Acquired by TechGiant Corp

3. InnovateTech
   - Market Share: 8.7%
   - Pricing: 10% below market
   - Strengths: Modern interface, fast deployment  
   - Weaknesses: Limited enterprise features
   - Recent News: Raised $50M Series B

MARKET TRENDS:
- AI integration becoming standard requirement
- Shift toward subscription-based pricing
- Increased focus on data security and compliance
- Growing demand for mobile accessibility

STRATEGIC RECOMMENDATIONS:
- Invest in AI capabilities to match SoftwareCorp
- Improve enterprise features to compete with market leaders
- Consider strategic acquisitions for quick market expansion
- Focus on unique value proposition for differentiation
""",
        
        "budget_proposal.txt": """2025 Budget Proposal
Department: Product Development

REQUESTED BUDGET: $3,200,000
Previous Year: $2,750,000
Increase: 16.4%

ALLOCATION BREAKDOWN:

Personnel (65% - $2,080,000):
- Software Engineers: $1,200,000 (8 engineers)
- Product Managers: $480,000 (3 managers)  
- UX/UI Designers: $300,000 (2 designers)
- QA Engineers: $100,000 (1 engineer)

Technology & Tools (20% - $640,000):
- Development Tools & Licenses: $200,000
- Cloud Infrastructure: $180,000
- Third-party APIs & Services: $120,000
- Hardware & Equipment: $140,000

Marketing & Operations (15% - $480,000):
- Product Marketing: $200,000
- User Research: $80,000
- Training & Conferences: $60,000
- Miscellaneous: $140,000

KEY INITIATIVES FOR 2025:
1. AI Integration Platform ($800,000)
2. Mobile Application Rewrite ($600,000)
3. Enterprise Security Features ($400,000)
4. API Platform Enhancement ($300,000)

EXPECTED OUTCOMES:
- 40% increase in development velocity
- Launch 3 major product features
- Improve customer satisfaction by 25%
- Capture additional $2M in annual revenue

ROI PROJECTION:
Investment: $3,200,000
Expected Additional Revenue: $2,000,000
Payback Period: 19.2 months
"""
    }
    
    # Create documents directory
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Write sample documents
    for filename, content in sample_docs.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"📁 Created {len(sample_docs)} sample documents in '{docs_dir}' directory")
    return list(sample_docs.keys())

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_file_processing_need():
    """Show why agents need file processing capabilities"""
    print("📄 WHY AGENTS NEED FILE PROCESSING")
    print("=" * 60)
    
    scenarios = [
        {
            "business_need": "Quarterly Report Analysis",
            "without_files": "Agent: 'I need you to manually input all the financial data'",
            "with_files": "Agent analyzes entire quarterly report automatically",
            "impact": "Hours saved, comprehensive analysis, no manual errors"
        },
        {
            "business_need": "Contract Review",
            "without_files": "Agent: 'Please copy and paste the contract terms'",
            "with_files": "Agent reviews full contract and identifies key terms",
            "impact": "Thorough analysis, risk identification, compliance checking"
        },
        {
            "business_need": "Competitive Intelligence",
            "without_files": "Agent: 'Describe what you know about competitors'",
            "with_files": "Agent analyzes competitor reports + web research",
            "impact": "Data-driven insights, comprehensive intelligence"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['business_need']}")
        print(f"   ❌ Without Files: {scenario['without_files']}")
        print(f"   ✅ With Files: {scenario['with_files']}")
        print(f"   💼 Business Impact: {scenario['impact']}")
    
    print("\n🎯 Key Insight: File processing transforms agents into comprehensive business analysts!")

def demonstrate_multi_tool_workflows():
    """Show powerful multi-tool analysis patterns"""
    print("\n🔧 MULTI-TOOL WORKFLOW PATTERNS")
    print("=" * 60)
    
    workflows = [
        {
            "pattern": "Document + Market Research",
            "steps": "1. Analyze internal report → 2. Research market trends → 3. Compare and insights",
            "example": "Quarterly results + industry benchmarks → competitive positioning"
        },
        {
            "pattern": "Financial Analysis + Calculations",
            "steps": "1. Process financial documents → 2. Calculate key metrics → 3. Research benchmarks",
            "example": "Budget proposal + ROI calculations + market salary data"
        },
        {
            "pattern": "Competitive Intelligence",
            "steps": "1. Analyze competitor files → 2. Web research updates → 3. Calculate market gaps",
            "example": "Competitor analysis + recent news + market share calculations"
        }
    ]
    
    for i, workflow in enumerate(workflows, 1):
        print(f"\n🔄 Pattern {i}: {workflow['pattern']}")
        print(f"   Steps: {workflow['steps']}")
        print(f"   Example: {workflow['example']}")
    
    print("\n🎯 Multi-tool workflows provide comprehensive business intelligence!")

# =============================================================================
# TESTING FILE PROCESSING
# =============================================================================

def test_file_processing_capabilities():
    """Test the multi-tool agent with file processing scenarios"""
    print("\n🧪 TESTING FILE PROCESSING CAPABILITIES")
    print("=" * 60)
    
    # Create sample documents first
    sample_files = create_sample_documents()
    
    agent = MultiToolAgent()
    
    test_cases = [
        {
            "name": "Financial Report Analysis",
            "question": f"Analyze our quarterly report in 'sample_documents/quarterly_report.txt' and research current software industry growth rates. How do we compare?",
            "expected_tools": "File analysis + web search + calculations"
        },
        {
            "name": "Competitive Intelligence",
            "question": f"Review our competitor analysis in 'sample_documents/competitor_analysis.txt' and find recent news about these companies. What strategic moves should we consider?",
            "expected_tools": "File analysis + web search for updates"
        },
        {
            "name": "Budget Analysis & Benchmarking",
            "question": f"Examine our budget proposal in 'sample_documents/budget_proposal.txt' and research market salary data for software engineers. Is our budget realistic?",
            "expected_tools": "File analysis + web research + salary calculations"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 File Processing Test {i}: {test['name']}")
        print(f"📄 Expected Tools: {test['expected_tools']}")
        print(f"❓ Question: {test['question']}")
        
        result = agent.comprehensive_analysis(test['question'])
        print(f"🏆 Multi-Tool Result: {result}")
        print("\n" + "=" * 80)
        
        if i < len(test_cases):
            input("Press Enter to continue to next file processing test...")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

def file_processing_workshop():
    """Interactive workshop with file processing capabilities"""
    print("\n🎯 FILE PROCESSING WORKSHOP CHALLENGE")
    print("=" * 60)
    
    agent = MultiToolAgent()
    
    print("Test your multi-tool agent with file processing + research scenarios!")
    print("Great challenges:")
    print("• Analyze business documents + market research")
    print("• Process financial reports + calculate benchmarks")
    print("• Review contracts + research legal standards")
    print("• Examine competitor data + find recent updates")
    print("\nSample files available in 'sample_documents/' directory")
    print("Type 'exit' to finish this quarter.")
    
    while True:
        user_question = input("\n💬 Your file processing + research question: ")
        
        if user_question.lower() in ['exit', 'quit', 'done']:
            print("🎉 Outstanding! Your agent can now process documents AND research online!")
            break
        
        if user_question.strip():
            result = agent.comprehensive_analysis(user_question)
            print(f"\n🎯 Multi-Tool Analysis Result: {result}")
        else:
            print("Please enter a question that involves file processing and/or research.")

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

def run_hour2_q2_workshop():
    """Main function for Hour 2 Q2 workshop"""
    print("🚀 HOUR 2 - QUARTER 2: FILE PROCESSING INTEGRATION")
    print("=" * 70)
    
    # Step 1: Show file processing need
    demonstrate_file_processing_need()
    
    # Step 2: Show multi-tool workflows
    demonstrate_multi_tool_workflows()
    
    # Step 3: Test file processing capabilities
    test_file_processing_capabilities()
    
    # Step 4: Interactive workshop
    file_processing_workshop()
    
    # Step 5: Quarter completion and Q3 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 2 COMPLETE!")
    print("=" * 60)
    print("File Processing Integration Achievements:")
    print("✅ Document analysis and intelligent summarization")
    print("✅ Multi-format file processing capabilities")
    print("✅ Combined file analysis + web research workflows")
    print("✅ Comprehensive business document intelligence")
    print("✅ Three-tool coordination (File + Search + Calculator)")
    
    print("\n🏆 Your Agent Can Now:")
    print("   → Analyze quarterly reports, budgets, and business documents")
    print("   → Combine internal document insights with external research")
    print("   → Process competitor analyses and market intelligence")
    print("   → Perform comprehensive financial and strategic analysis")
    
    print("\n📈 Tool Evolution Completed:")
    print("   Hour 1: Calculator → Reasoning + Basic Tools")
    print("   Hour 2 Q1: + Web Search → Internet Research")
    print("   Hour 2 Q2: + File Processing → Document Intelligence")
    print("   Hour 2 Q3: Multi-Tool Coordination → Intelligent Workflows")
    
    print("\n🚀 Coming Up in Q3: Multi-Tool Coordination")
    print("   → Intelligent tool chaining and workflow optimization")
    print("   → Advanced reasoning about which tools to use when")
    print("   → Seamless integration of all capabilities")
    print("   → Production-ready multi-tool automation")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 2 Q3: Multi-Tool Coordination!")

if __name__ == "__main__":
    # Run the complete Hour 2 Q2 workshop
    run_hour2_q2_workshop()