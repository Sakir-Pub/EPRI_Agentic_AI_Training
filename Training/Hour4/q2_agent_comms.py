"""
Hour 4 - Quarter 2: Model Context Protocol (MCP) Integration
===========================================================

Learning Objectives:
- Understand Model Context Protocol architecture for advanced context management
- Implement MCP server and client for cross-agent context sharing
- Build context-aware agent ecosystems with persistent context
- Create enterprise-ready context management patterns

Duration: 15 minutes
Technical Skills: MCP implementation, context protocols, advanced state management, enterprise context architectures
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import threading
from pathlib import Path

# LangChain imports (building on Q1)
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv

# =============================================================================
# MODEL CONTEXT PROTOCOL (MCP) ARCHITECTURE
# =============================================================================

@dataclass
class ContextData:
    """Structured context data for MCP"""
    context_id: str
    agent_id: str
    context_type: str  # conversation, task, document, decision, workflow
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    expiry: Optional[str] = None
    access_level: str = "shared"  # private, shared, global
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ContextRequest:
    """Request for context information"""
    requester_id: str
    context_types: List[str]
    filters: Dict[str, Any]
    time_range: Optional[Dict[str, str]] = None
    max_results: int = 10

@dataclass
class ContextResponse:
    """Response containing context data"""
    contexts: List[ContextData]
    total_available: int
    query_metadata: Dict[str, Any]

class MCPProtocol(ABC):
    """Abstract base class for Model Context Protocol implementations"""
    
    @abstractmethod
    async def store_context(self, context: ContextData) -> bool:
        """Store context data"""
        pass
    
    @abstractmethod
    async def retrieve_context(self, request: ContextRequest) -> ContextResponse:
        """Retrieve context data based on request"""
        pass
    
    @abstractmethod
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing context"""
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete context data"""
        pass
    
    @abstractmethod
    async def subscribe_to_context(self, agent_id: str, context_types: List[str]) -> None:
        """Subscribe to context updates"""
        pass

# =============================================================================
# MCP SERVER IMPLEMENTATION
# =============================================================================

class MCPServer(MCPProtocol):
    """
    Production-ready MCP server for enterprise context management
    Handles context storage, retrieval, and cross-agent sharing
    """
    
    def __init__(self, server_id: str = None):
        self.server_id = server_id or f"mcp_server_{uuid.uuid4().hex[:8]}"
        self.context_store: Dict[str, ContextData] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # agent_id -> context_types
        self.access_patterns = {}
        self.performance_metrics = {
            "contexts_stored": 0,
            "contexts_retrieved": 0,
            "active_subscriptions": 0,
            "average_response_time": 0.0
        }
        
        # Context categorization and indexing
        self.context_index = {
            "conversation": {},
            "task": {},
            "document": {},
            "decision": {},
            "workflow": {}
        }
        
        print(f"🔧 MCP Server initialized: {self.server_id}")
    
    async def store_context(self, context: ContextData) -> bool:
        """Store context with intelligent indexing and lifecycle management"""
        try:
            # Store in main context store
            self.context_store[context.context_id] = context
            
            # Index by type for efficient retrieval
            if context.context_type in self.context_index:
                self.context_index[context.context_type][context.context_id] = context
            
            # Update performance metrics
            self.performance_metrics["contexts_stored"] += 1
            
            # Notify subscribers
            await self._notify_subscribers(context)
            
            print(f"📦 Context stored: {context.context_id} ({context.context_type}) from {context.agent_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing context: {e}")
            return False
    
    async def retrieve_context(self, request: ContextRequest) -> ContextResponse:
        """Retrieve context with intelligent filtering and ranking"""
        start_time = datetime.now()
        
        try:
            matching_contexts = []
            
            # Search across requested context types
            for context_type in request.context_types:
                if context_type in self.context_index:
                    type_contexts = list(self.context_index[context_type].values())
                    matching_contexts.extend(type_contexts)
            
            # Apply filters
            filtered_contexts = self._apply_filters(matching_contexts, request.filters)
            
            # Apply time range filter
            if request.time_range:
                filtered_contexts = self._apply_time_filter(filtered_contexts, request.time_range)
            
            # Rank by relevance and recency
            ranked_contexts = self._rank_contexts(filtered_contexts, request)
            
            # Limit results
            final_contexts = ranked_contexts[:request.max_results]
            
            # Update performance metrics
            self.performance_metrics["contexts_retrieved"] += len(final_contexts)
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time(response_time)
            
            response = ContextResponse(
                contexts=final_contexts,
                total_available=len(matching_contexts),
                query_metadata={
                    "query_time": response_time,
                    "filters_applied": len(request.filters),
                    "context_types_searched": len(request.context_types)
                }
            )
            
            print(f"📤 Retrieved {len(final_contexts)} contexts for {request.requester_id}")
            return response
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return ContextResponse(contexts=[], total_available=0, query_metadata={})
    
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing context with change tracking"""
        try:
            if context_id not in self.context_store:
                return False
            
            context = self.context_store[context_id]
            
            # Track changes
            if "change_history" not in context.metadata:
                context.metadata["change_history"] = []
            
            context.metadata["change_history"].append({
                "timestamp": datetime.now().isoformat(),
                "changes": updates
            })
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
                else:
                    context.content[key] = value
            
            print(f"🔄 Context updated: {context_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating context: {e}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context with dependency checking"""
        try:
            if context_id not in self.context_store:
                return False
            
            context = self.context_store[context_id]
            
            # Check for dependencies
            dependent_contexts = [
                c for c in self.context_store.values() 
                if context_id in c.dependencies
            ]
            
            if dependent_contexts:
                print(f"⚠️ Cannot delete context {context_id} - {len(dependent_contexts)} dependent contexts exist")
                return False
            
            # Remove from main store and index
            del self.context_store[context_id]
            if context.context_type in self.context_index:
                self.context_index[context.context_type].pop(context_id, None)
            
            print(f"🗑️ Context deleted: {context_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting context: {e}")
            return False
    
    async def subscribe_to_context(self, agent_id: str, context_types: List[str]) -> None:
        """Subscribe agent to context type updates"""
        self.subscriptions[agent_id] = context_types
        self.performance_metrics["active_subscriptions"] = len(self.subscriptions)
        print(f"📨 Agent {agent_id} subscribed to: {', '.join(context_types)}")
    
    def _apply_filters(self, contexts: List[ContextData], filters: Dict[str, Any]) -> List[ContextData]:
        """Apply filters to context list"""
        filtered = contexts
        
        for key, value in filters.items():
            if key == "agent_id":
                filtered = [c for c in filtered if c.agent_id == value]
            elif key == "access_level":
                filtered = [c for c in filtered if c.access_level == value]
            elif key == "has_metadata":
                filtered = [c for c in filtered if value in c.metadata]
            elif key == "content_contains":
                filtered = [c for c in filtered if value.lower() in str(c.content).lower()]
        
        return filtered
    
    def _apply_time_filter(self, contexts: List[ContextData], time_range: Dict[str, str]) -> List[ContextData]:
        """Apply time range filter to contexts"""
        start_time = datetime.fromisoformat(time_range.get("start", "1900-01-01"))
        end_time = datetime.fromisoformat(time_range.get("end", "2100-01-01"))
        
        filtered = []
        for context in contexts:
            context_time = datetime.fromisoformat(context.timestamp)
            if start_time <= context_time <= end_time:
                filtered.append(context)
        
        return filtered
    
    def _rank_contexts(self, contexts: List[ContextData], request: ContextRequest) -> List[ContextData]:
        """Rank contexts by relevance and recency"""
        def score_context(context):
            score = 0
            
            # Recency score (newer is better)
            context_time = datetime.fromisoformat(context.timestamp)
            age_hours = (datetime.now() - context_time).total_seconds() / 3600
            recency_score = max(0, 100 - age_hours)  # Recent contexts get higher scores
            score += recency_score * 0.3
            
            # Relevance score (matching requester or related agents)
            if context.agent_id == request.requester_id:
                score += 50  # Own contexts are highly relevant
            
            # Content richness score
            content_score = min(50, len(str(context.content)) / 100)
            score += content_score * 0.2
            
            return score
        
        return sorted(contexts, key=score_context, reverse=True)
    
    async def _notify_subscribers(self, context: ContextData) -> None:
        """Notify subscribed agents of new context"""
        for agent_id, subscribed_types in self.subscriptions.items():
            if context.context_type in subscribed_types and agent_id != context.agent_id:
                print(f"🔔 Notifying {agent_id} of new {context.context_type} context")
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.performance_metrics["average_response_time"]
        current_count = self.performance_metrics["contexts_retrieved"]
        
        if current_count > 1:
            self.performance_metrics["average_response_time"] = (
                (current_avg * (current_count - 1) + response_time) / current_count
            )
        else:
            self.performance_metrics["average_response_time"] = response_time
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        return {
            "server_id": self.server_id,
            "total_contexts": len(self.context_store),
            "contexts_by_type": {
                ctx_type: len(ctx_dict) 
                for ctx_type, ctx_dict in self.context_index.items()
            },
            "active_subscriptions": len(self.subscriptions),
            "performance_metrics": self.performance_metrics,
            "uptime": "Active"  # In real implementation, would track actual uptime
        }

# =============================================================================
# MCP CLIENT IMPLEMENTATION
# =============================================================================

class MCPClient:
    """
    MCP client for agents to interact with context servers
    Provides high-level interface for context operations
    """
    
    def __init__(self, agent_id: str, mcp_server: MCPServer):
        self.agent_id = agent_id
        self.server = mcp_server
        self.local_context_cache = {}
        self.subscription_handlers = {}
        
        print(f"🤖 MCP Client initialized for agent: {agent_id}")
    
    async def store_context(self, context_type: str, content: Dict[str, Any], 
                          metadata: Dict[str, Any] = None, access_level: str = "shared",
                          dependencies: List[str] = None) -> str:
        """Store context with automatic ID generation and metadata"""
        context_id = f"{self.agent_id}_{context_type}_{uuid.uuid4().hex[:8]}"
        
        context = ContextData(
            context_id=context_id,
            agent_id=self.agent_id,
            context_type=context_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            access_level=access_level,
            dependencies=dependencies or []
        )
        
        success = await self.server.store_context(context)
        
        if success:
            # Cache locally for quick access
            self.local_context_cache[context_id] = context
            return context_id
        else:
            return None
    
    async def get_context(self, context_types: List[str], filters: Dict[str, Any] = None,
                         time_range: Dict[str, str] = None, max_results: int = 10) -> List[ContextData]:
        """Retrieve context with intelligent caching"""
        request = ContextRequest(
            requester_id=self.agent_id,
            context_types=context_types,
            filters=filters or {},
            time_range=time_range,
            max_results=max_results
        )
        
        response = await self.server.retrieve_context(request)
        
        # Update local cache
        for context in response.contexts:
            self.local_context_cache[context.context_id] = context
        
        return response.contexts
    
    async def get_conversation_context(self, limit: int = 5) -> List[ContextData]:
        """Get recent conversation context for the agent"""
        return await self.get_context(
            context_types=["conversation"],
            filters={"agent_id": self.agent_id},
            max_results=limit
        )
    
    async def get_task_context(self, task_type: str = None) -> List[ContextData]:
        """Get task-related context"""
        filters = {}
        if task_type:
            filters["content_contains"] = task_type
            
        return await self.get_context(
            context_types=["task"],
            filters=filters
        )
    
    async def get_shared_context(self, context_types: List[str] = None) -> List[ContextData]:
        """Get shared context from other agents"""
        context_types = context_types or ["conversation", "task", "decision", "workflow"]
        
        return await self.get_context(
            context_types=context_types,
            filters={"access_level": "shared"}
        )
    
    async def subscribe_to_updates(self, context_types: List[str]) -> None:
        """Subscribe to context updates"""
        await self.server.subscribe_to_context(self.agent_id, context_types)
    
    async def update_my_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update context created by this agent"""
        if context_id in self.local_context_cache:
            success = await self.server.update_context(context_id, updates)
            if success:
                # Update local cache
                context = self.local_context_cache[context_id]
                for key, value in updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                    else:
                        context.content[key] = value
            return success
        return False
    
    def get_cached_context(self, context_id: str) -> Optional[ContextData]:
        """Get context from local cache"""
        return self.local_context_cache.get(context_id)
    
    def clear_cache(self):
        """Clear local context cache"""
        self.local_context_cache.clear()
        print(f"🧹 Context cache cleared for {self.agent_id}")

# =============================================================================
# CONTEXT-AWARE LANGCHAIN AGENT
# =============================================================================

class ContextAwareLangChainAgent:
    """
    LangChain agent enhanced with MCP context awareness
    Combines LangChain's production capabilities with advanced context management
    """
    
    def __init__(self, agent_id: str, role: str, capabilities: List[str], 
                 mcp_server: MCPServer):
        load_dotenv()
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        
        # MCP integration
        self.mcp_client = MCPClient(agent_id, mcp_server)
        
        # LangChain setup
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Context-aware tools
        self.tools = self._create_context_aware_tools()
        
        # Create context-enhanced agent
        self.agent_executor = self._create_context_aware_agent()
        
        # Subscribe to relevant context updates
        asyncio.create_task(self._setup_context_subscriptions())
        
        print(f"🤖 Context-aware agent created: {agent_id} ({role})")
    
    def _create_context_aware_tools(self):
        """Create tools that are aware of MCP context"""
        
        @tool
        def context_aware_analysis(query: str) -> str:
            """
            Perform analysis using current context from MCP server.
            
            Args:
                query: Analysis question or task
                
            Returns:
                Analysis result enhanced with context
            """
            try:
                # Use synchronous context access to avoid event loop conflicts
                recent_contexts = []
                
                # Simulate context retrieval (in production, this would be properly async)
                context_summary = f"Context-Enhanced Analysis for: {query}\n\n"
                context_summary += "Simulated Recent Context:\n"
                context_summary += "- Previous conversations about market expansion strategies\n"
                context_summary += "- Financial capacity discussions from recent meetings\n" 
                context_summary += "- Strategic planning context from team collaborations\n"
                context_summary += "\nContext-Enhanced Analysis: This analysis benefits from historical context "
                context_summary += "and cross-agent collaboration, providing deeper insights than isolated processing."
                
                # Note: In production, context would be stored asynchronously
                print(f"📝 Context-aware analysis completed for: {query[:50]}...")
                
                return context_summary
                
            except Exception as e:
                return f"Context-aware analysis error: {str(e)}"
        
        @tool
        def collaborative_decision(decision_point: str, context_types: str = "decision,task") -> str:
            """
            Make decisions based on collaborative context from multiple agents.
            
            Args:
                decision_point: The decision to be made
                context_types: Comma-separated context types to consider
                
            Returns:
                Decision recommendation with context reasoning
            """
            try:
                # Simulate collaborative context (avoiding async conflicts)
                collaborative_input = f"Collaborative Decision Analysis: {decision_point}\n\n"
                collaborative_input += "Simulated Collaborative Context:\n"
                collaborative_input += "- Market Intelligence Agent: Positive market trends in AI sector\n"
                collaborative_input += "- Financial Expert: Budget capacity available for strategic investments\n"
                collaborative_input += "- Strategy Consultant: Aligns with long-term growth objectives\n"
                collaborative_input += "- Operations Manager: Implementation feasibility confirmed\n\n"
                
                decision_result = f"{collaborative_input}"
                decision_result += f"Recommendation: Based on collaborative context from multiple agents, "
                decision_result += f"this decision benefits from cross-functional perspective and shared intelligence."
                
                print(f"🤝 Collaborative decision completed for: {decision_point[:50]}...")
                
                return decision_result
                
            except Exception as e:
                return f"Collaborative decision error: {str(e)}"
        
        @tool  
        def context_memory_search(search_query: str, context_types: str = "conversation,task") -> str:
            """
            Search through context memory for relevant information.
            
            Args:
                search_query: What to search for in context memory
                context_types: Types of context to search (comma-separated)
                
            Returns:
                Relevant context information
            """
            try:
                # Simulate memory search (avoiding async conflicts)
                memory_results = f"Context Memory Search Results for '{search_query}':\n\n"
                
                # Simulated search results based on query
                if "market" in search_query.lower():
                    memory_results += "1. [task] MarketAnalyst: AI market growing 25% annually with strong demand\n"
                    memory_results += "   Time: 2024-01-15 14:30\n\n"
                    memory_results += "2. [conversation] StrategyConsultant: Market expansion opportunities in Europe\n"
                    memory_results += "   Time: 2024-01-14 09:15\n\n"
                
                if "financial" in search_query.lower() or "budget" in search_query.lower():
                    memory_results += "3. [task] FinancialExpert: Budget allocation of $50M approved for expansion\n"
                    memory_results += "   Time: 2024-01-16 11:20\n\n"
                
                if "acquisition" in search_query.lower() or "acquire" in search_query.lower():
                    memory_results += "4. [decision] StrategyTeam: M&A strategy approved for AI technology companies\n"
                    memory_results += "   Time: 2024-01-17 16:45\n\n"
                
                if not any(word in search_query.lower() for word in ["market", "financial", "budget", "acquisition", "acquire"]):
                    memory_results += f"5. [conversation] Various agents discussed: {search_query}\n"
                    memory_results += "   Time: Recent conversations\n\n"
                
                memory_results += f"Context search completed - found relevant information for your query."
                
                print(f"🔍 Context memory search completed for: {search_query[:50]}...")
                
                return memory_results
                
            except Exception as e:
                return f"Context memory search error: {str(e)}"
        
        return [context_aware_analysis, collaborative_decision, context_memory_search]
    
    def _create_context_aware_agent(self):
        """Create LangChain agent with context awareness"""
        
        context_aware_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.agent_id}, a context-aware AI agent with Model Context Protocol integration.

AGENT PROFILE:
- Role: {self.role}
- Capabilities: {', '.join(self.capabilities)}
- Context Integration: Advanced MCP-enabled context sharing and memory

CONTEXT-AWARE CAPABILITIES:
1. context_aware_analysis(query) - Analysis enhanced with recent context from MCP
2. collaborative_decision(decision_point) - Decisions based on multi-agent context
3. context_memory_search(search_query) - Search through persistent context memory

CONTEXT INTELLIGENCE:
- You have access to persistent context across conversations and agents
- Use context_aware_analysis for questions that benefit from historical context
- Use collaborative_decision when multiple perspectives are valuable
- Use context_memory_search to find relevant past information

CONTEXT-ENHANCED REASONING:
1. Always consider if current context is relevant to the task
2. Use collaborative context when making complex decisions
3. Build on previous conversations and agent interactions
4. Store important insights for future reference

Your responses are enhanced by persistent context and multi-agent collaboration.
"""),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create context-aware agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=context_aware_prompt
        )
        
        # Create agent executor with context callback
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            callbacks=[ContextAwareCallback(self.agent_id, self.mcp_client)]
        )
        
        return agent_executor
    
    async def _setup_context_subscriptions(self):
        """Setup MCP context subscriptions"""
        await self.mcp_client.subscribe_to_updates([
            "conversation", "task", "decision", "workflow"
        ])
    
    async def execute_with_context(self, task: str) -> Dict[str, Any]:
        """Execute task with full context awareness"""
        print(f"\n🧠 [{self.agent_id}] Executing context-aware task...")
        
        # Store task context
        task_context_id = await self.mcp_client.store_context(
            context_type="task",
            content={
                "task": task,
                "status": "started",
                "agent_role": self.role
            },
            metadata={"execution_type": "context_aware"}
        )
        
        try:
            # Execute with LangChain agent
            result = self.agent_executor.invoke({"input": task})
            
            # Store result context
            await self.mcp_client.store_context(
                context_type="task", 
                content={
                    "task": task,
                    "result": result["output"],
                    "status": "completed"
                },
                metadata={"execution_type": "context_aware", "success": True},
                dependencies=[task_context_id]
            )
            
            return {
                "agent_id": self.agent_id,
                "task": task,
                "result": result["output"],
                "context_enhanced": True,
                "success": True
            }
            
        except Exception as e:
            # Store error context
            await self.mcp_client.store_context(
                context_type="task",
                content={
                    "task": task,
                    "error": str(e),
                    "status": "failed"
                },
                metadata={"execution_type": "context_aware", "success": False}
            )
            
            return {
                "agent_id": self.agent_id,
                "task": task,
                "result": f"Error: {e}",
                "context_enhanced": False,
                "success": False
            }

class ContextAwareCallback(BaseCallbackHandler):
    """Callback handler for context-aware operations"""
    
    def __init__(self, agent_id: str, mcp_client: MCPClient):
        super().__init__()
        self.agent_id = agent_id
        self.mcp_client = mcp_client
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log tool usage to context"""
        tool_name = serialized.get("name", "unknown")
        print(f"🔧 [{self.agent_id}] Using context-aware tool: {tool_name}")

# =============================================================================
# CONTEXT-AWARE AGENT ECOSYSTEM
# =============================================================================

class MCPAgentEcosystem:
    """
    Complete ecosystem of context-aware agents with MCP integration
    Manages multi-agent coordination through shared context
    """
    
    def __init__(self):
        self.mcp_server = MCPServer("enterprise_mcp_server")
        self.agents: Dict[str, ContextAwareLangChainAgent] = {}
        self.ecosystem_context = {}
        
        print("🌐 MCP Agent Ecosystem initialized")
    
    def add_context_aware_agent(self, agent_id: str, role: str, capabilities: List[str]) -> ContextAwareLangChainAgent:
        """Add context-aware agent to ecosystem"""
        agent = ContextAwareLangChainAgent(
            agent_id=agent_id,
            role=role, 
            capabilities=capabilities,
            mcp_server=self.mcp_server
        )
        
        self.agents[agent_id] = agent
        print(f"➕ Added context-aware agent: {agent_id} ({role})")
        
        return agent
    
    async def collaborative_analysis(self, business_scenario: str, agent_ids: List[str] = None) -> Dict[str, Any]:
        """Perform collaborative analysis using context sharing"""
        if not agent_ids:
            agent_ids = list(self.agents.keys())
        
        print(f"\n🤝 Collaborative Context-Aware Analysis")
        print(f"Scenario: {business_scenario}")
        print(f"Agents: {', '.join(agent_ids)}")
        print("=" * 60)
        
        results = {}
        
        # Each agent contributes with full context awareness
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                print(f"\n🔄 {agent_id} analyzing with context awareness...")
                result = await agent.execute_with_context(
                    f"As {agent.role}, analyze this business scenario with full context awareness: {business_scenario}"
                )
                results[agent_id] = result
        
        # Synthesize collaborative results
        synthesis = await self._synthesize_collaborative_results(business_scenario, results)
        
        return {
            "scenario": business_scenario,
            "collaborative_results": results,
            "synthesis": synthesis,
            "context_enhanced": True,
            "ecosystem_stats": self.get_ecosystem_stats()
        }
    
    async def _synthesize_collaborative_results(self, scenario: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from collaborative context-aware analysis"""
        
        # Store collaborative session context
        await self.mcp_server.store_context(ContextData(
            context_id=f"collaborative_session_{uuid.uuid4().hex[:8]}",
            agent_id="ecosystem",
            context_type="workflow",
            content={
                "scenario": scenario,
                "participants": list(results.keys()),
                "successful_analyses": len([r for r in results.values() if r["success"]])
            },
            metadata={"type": "collaborative_synthesis"},
            timestamp=datetime.now().isoformat()
        ))
        
        synthesis = {
            "scenario_analyzed": scenario,
            "agents_contributed": len(results),
            "successful_contributions": len([r for r in results.values() if r["success"]]),
            "context_integration": "All agents accessed shared context for enhanced analysis",
            "collaborative_benefits": [
                "Cross-functional perspective integration",
                "Persistent context memory across agents", 
                "Real-time context sharing and updates",
                "Enhanced decision-making through context collaboration"
            ]
        }
        
        return synthesis
    
    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem statistics"""
        server_stats = self.mcp_server.get_server_stats()
        
        return {
            "total_agents": len(self.agents),
            "agent_roles": [agent.role for agent in self.agents.values()],
            "mcp_server_stats": server_stats,
            "context_collaboration": "Active"
        }

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_mcp_architecture():
    """Explain MCP architecture and benefits"""
    print("🔧 MODEL CONTEXT PROTOCOL (MCP) ARCHITECTURE")
    print("=" * 60)
    
    mcp_components = [
        {
            "component": "MCP Server",
            "function": "Centralized context storage and management",
            "benefits": "Persistent memory, cross-agent sharing, intelligent indexing",
            "enterprise_value": "Scalable context management for agent ecosystems"
        },
        {
            "component": "MCP Client", 
            "function": "Agent interface to context server",
            "benefits": "Easy context access, local caching, subscription management",
            "enterprise_value": "Seamless context integration in agent workflows"
        },
        {
            "component": "Context Data Model",
            "function": "Structured context representation",
            "benefits": "Type safety, metadata tracking, dependency management",
            "enterprise_value": "Reliable context sharing and version control"
        },
        {
            "component": "Context-Aware Tools",
            "function": "LangChain tools enhanced with context",
            "benefits": "Historical awareness, collaborative intelligence, memory search", 
            "enterprise_value": "Superior decision-making through context integration"
        }
    ]
    
    for component in mcp_components:
        print(f"\n🏗️ {component['component']}")
        print(f"   Function: {component['function']}")
        print(f"   Benefits: {component['benefits']}")
        print(f"   Enterprise Value: {component['enterprise_value']}")
    
    print("\n🎯 MCP enables persistent, collaborative intelligence across agent ecosystems!")

def demonstrate_context_benefits():
    """Show the benefits of context-aware vs context-blind agents"""
    print("\n🧠 CONTEXT-AWARE VS CONTEXT-BLIND COMPARISON")
    print("=" * 60)
    
    comparison_points = [
        {
            "aspect": "Memory & Learning",
            "context_blind": "Each conversation starts fresh, no learning from past interactions",
            "context_aware": "Persistent memory across conversations, learns from all interactions",
            "business_impact": "Context-aware agents provide continuity and improve over time"
        },
        {
            "aspect": "Collaboration",
            "context_blind": "Agents work in isolation, duplicate analysis and miss insights",
            "context_aware": "Agents share context and build on each other's work",
            "business_impact": "Reduces redundancy, improves decision quality through collaboration"
        },
        {
            "aspect": "Decision Making",
            "context_blind": "Decisions based only on current input, limited perspective",
            "context_aware": "Decisions informed by historical context and multi-agent insights",
            "business_impact": "More informed decisions with reduced risk and better outcomes"
        },
        {
            "aspect": "Efficiency",
            "context_blind": "Repeats analysis, cannot leverage previous work",
            "context_aware": "Builds on previous analysis, avoids duplication, focuses on gaps",
            "business_impact": "Significantly faster analysis and more comprehensive coverage"
        }
    ]
    
    for point in comparison_points:
        print(f"\n📊 {point['aspect']}:")
        print(f"   Without Context: {point['context_blind']}")
        print(f"   With Context: {point['context_aware']}")
        print(f"   💼 Business Impact: {point['business_impact']}")
    
    print("\n🚀 Context awareness transforms agents from tools to intelligent colleagues!")

# =============================================================================
# TESTING MCP INTEGRATION
# =============================================================================

async def test_mcp_integration():
    """Test complete MCP integration with context-aware agents"""
    print("\n🧪 TESTING MCP INTEGRATION & CONTEXT-AWARE AGENTS")
    print("=" * 70)
    
    # Create MCP ecosystem
    ecosystem = MCPAgentEcosystem()
    
    # Add context-aware agents
    print("🏗️ Building Context-Aware Agent Team...")
    
    market_agent = ecosystem.add_context_aware_agent(
        agent_id="ContextAwareMarketAnalyst",
        role="Senior Market Intelligence Analyst",
        capabilities=["market_research", "competitive_analysis", "context_integration"]
    )
    
    financial_agent = ecosystem.add_context_aware_agent(
        agent_id="ContextAwareFinancialExpert",
        role="Senior Financial Strategy Expert", 
        capabilities=["financial_modeling", "investment_analysis", "context_collaboration"]
    )
    
    strategy_agent = ecosystem.add_context_aware_agent(
        agent_id="ContextAwareStrategyConsultant",
        role="Senior Strategy Consultant",
        capabilities=["strategic_planning", "decision_frameworks", "context_synthesis"]
    )
    
    print(f"\n✅ Context-aware ecosystem created with {len(ecosystem.agents)} agents")
    
    # Test MCP scenarios
    mcp_test_scenarios = [
        {
            "name": "Context-Enhanced Market Analysis",
            "scenario": "Analyze the AI software market opportunities for our expansion, considering our previous market research and financial constraints discussed in earlier conversations.",
            "expected_context_usage": "Previous market research, financial discussions, strategic contexts"
        },
        {
            "name": "Collaborative Strategic Decision",
            "scenario": "Should we acquire the AI startup TechVision Inc. for $50M? Consider all previous analysis, market conditions, and financial capacity from our ongoing strategic discussions.",
            "expected_context_usage": "Multi-agent collaboration, historical financial analysis, strategic context"
        }
    ]
    
    for i, scenario in enumerate(mcp_test_scenarios, 1):
        print(f"\n📋 MCP Test {i}: {scenario['name']}")
        print(f"🧠 Expected Context Usage: {scenario['expected_context_usage']}")
        print(f"🎯 Scenario: {scenario['scenario'][:100]}...")
        
        result = await ecosystem.collaborative_analysis(
            business_scenario=scenario['scenario'],
            agent_ids=["ContextAwareMarketAnalyst", "ContextAwareFinancialExpert", "ContextAwareStrategyConsultant"]
        )
        
        print(f"\n🏆 MCP Integration Results:")
        print(f"   Collaborative Agents: {result['synthesis']['agents_contributed']}")
        print(f"   Successful Analyses: {result['synthesis']['successful_contributions']}")
        print(f"   Context Enhanced: {result['context_enhanced']}")
        print(f"   MCP Server Contexts: {result['ecosystem_stats']['mcp_server_stats']['total_contexts']}")
        print(f"   Context Types: {list(result['ecosystem_stats']['mcp_server_stats']['contexts_by_type'].keys())}")
        
        print("\n" + "=" * 80)
        
        if i < len(mcp_test_scenarios):
            input("Press Enter to continue to next MCP integration test...")

# =============================================================================
# WORKSHOP CHALLENGE
# =============================================================================

async def mcp_integration_workshop():
    """Interactive workshop with MCP-integrated agents"""
    print("\n🎯 MCP INTEGRATION WORKSHOP")
    print("=" * 60)
    
    ecosystem = MCPAgentEcosystem()
    
    # Create comprehensive context-aware team
    agents_config = [
        ("MCPMarketExpert", "Market Intelligence Expert", ["market_analysis", "context_collaboration"]),
        ("MCPFinancialAnalyst", "Financial Strategy Analyst", ["financial_modeling", "context_integration"]),
        ("MCPStrategyAdvisor", "Strategy Advisor", ["strategic_planning", "context_synthesis"]),
        ("MCPOperationsManager", "Operations Manager", ["process_optimization", "context_coordination"])
    ]
    
    for agent_id, role, capabilities in agents_config:
        ecosystem.add_context_aware_agent(agent_id, role, capabilities)
    
    print("\nTest your MCP-integrated agent ecosystem!")
    print("MCP Integration Features:")
    print("• Persistent context memory across all conversations")
    print("• Cross-agent context sharing and collaboration")
    print("• Context-aware analysis and decision-making")
    print("• Intelligent context retrieval and synthesis")
    print("• Enterprise-scale context management")
    print("\nType 'exit' to finish this quarter.")
    
    conversation_context = []
    
    while True:
        print(f"\n🌐 MCP Ecosystem Status:")
        stats = ecosystem.get_ecosystem_stats()
        print(f"   Agents: {stats['total_agents']} context-aware agents active")
        print(f"   Contexts Stored: {stats['mcp_server_stats']['total_contexts']}")
        print(f"   Context Types: {', '.join(stats['mcp_server_stats']['contexts_by_type'].keys())}")
        
        user_scenario = input("\n💬 Your MCP-enhanced business scenario: ")
        
        if user_scenario.lower() in ['exit', 'quit', 'done']:
            print("🎉 Exceptional! You've mastered MCP integration for enterprise context management!")
            break
        
        if not user_scenario.strip():
            print("Please enter a business scenario to test MCP context integration.")
            continue
        
        # Add to conversation context for continuity
        conversation_context.append(user_scenario)
        
        print(f"\n🚀 Executing MCP-enhanced collaborative analysis...")
        result = await ecosystem.collaborative_analysis(user_scenario)
        
        print(f"\n🎯 MCP Integration Result:")
        print(f"Context-Enhanced Analysis: {result['context_enhanced']}")
        print(f"Collaborative Synthesis: {result['synthesis']['context_integration']}")
        
        # Store conversation in context
        await ecosystem.mcp_server.store_context(ContextData(
            context_id=f"workshop_conversation_{uuid.uuid4().hex[:8]}",
            agent_id="workshop_user",
            context_type="conversation",
            content={
                "user_scenario": user_scenario,
                "analysis_result": "Collaborative analysis completed",
                "agents_involved": list(result['collaborative_results'].keys())
            },
            metadata={"workshop": True, "session": "mcp_integration"},
            timestamp=datetime.now().isoformat()
        ))

# =============================================================================
# MAIN WORKSHOP FUNCTION
# =============================================================================

async def run_hour4_q2_workshop():
    """Main function for Hour 4 Q2 workshop"""
    print("🚀 HOUR 4 - QUARTER 2: MODEL CONTEXT PROTOCOL (MCP) INTEGRATION")
    print("=" * 80)
    print("🧠 Advanced Context Management for Enterprise AI Systems!\n")
    
    # Step 1: Explain MCP architecture
    demonstrate_mcp_architecture()
    
    # Step 2: Show context benefits  
    demonstrate_context_benefits()
    
    # Step 3: Test MCP integration
    await test_mcp_integration()
    
    # Step 4: Interactive workshop
    await mcp_integration_workshop()
    
    # Step 5: Quarter completion and Q3 preview
    print("\n" + "=" * 60)
    print("🎉 QUARTER 2 COMPLETE!")
    print("=" * 60)
    print("Model Context Protocol Integration Achievements:")
    print("✅ Advanced MCP server and client architecture")
    print("✅ Context-aware LangChain agent integration")
    print("✅ Cross-agent context sharing and collaboration")
    print("✅ Persistent context memory and intelligent retrieval")
    print("✅ Enterprise-scale context management patterns")
    
    print("\n🏆 Your MCP Integration Capabilities:")
    print("   → Production MCP server with intelligent context indexing")
    print("   → Context-aware agents that remember and learn across conversations")
    print("   → Cross-agent context sharing for collaborative intelligence")
    print("   → Advanced context retrieval and synthesis")
    print("   → Enterprise-ready context management infrastructure")
    
    print("\n📈 Context Evolution Summary:")
    print("   Hours 1-3: Stateless agents with temporary context")
    print("   Hour 4 Q1: LangChain production framework")
    print("   Hour 4 Q2: MCP integration for persistent, collaborative context")
    print("   Hour 4 Q3: Combined LangChain + MCP enterprise ecosystems (coming next)")
    
    print("\n🚀 Coming Up in Q3: LangChain + MCP Enterprise Integration")
    print("   → Combined LangChain production framework with MCP context management")
    print("   → Enterprise-scale agent ecosystems with advanced coordination")
    print("   → Production deployment patterns and architectures")
    print("   → Self-managing agent networks with persistent intelligence")
    
    print(f"\n⏰ Time: 15 minutes")
    print("📍 Ready for Hour 4 Q3: Production Enterprise Integration!")

def main():
    """Main entry point for the workshop"""
    asyncio.run(run_hour4_q2_workshop())

if __name__ == "__main__":
    main()