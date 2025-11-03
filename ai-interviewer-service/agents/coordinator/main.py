import asyncio
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import uuid

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import AIMessage as CoreAIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Database for problem management
import asyncpg

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import (
    AgentMessage, EventType, InterviewContext, InterviewState,
    Problem, Difficulty, Message, PerformanceMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangGraph State Schema
class InterviewWorkflowState(TypedDict):
    """State for the interview coordination workflow"""
    session_id: str
    user_id: str
    current_action: str
    user_input: Optional[str]
    interview_state: str
    current_problem: Optional[Dict[str, Any]]
    routing_decision: Optional[str]
    messages: List[BaseMessage]
    context: Dict[str, Any]
    next_step: Optional[str]

class CoordinatorAgent:
    """
    Master orchestrator using LangGraph for multi-agent coordination.
    Manages interview flow, routes messages, and controls state transitions.
    """

    def __init__(self):
        self.agent_name = "coordinator"
        self.message_bus: MessageBus = None
        self.db_pool: asyncpg.Pool = None
        self.running = False

        # LangChain components
        self.llm: ChatAnthropic = None
        self.memory: ConversationBufferWindowMemory = None

        # LangGraph components
        self.workflow: StateGraph = None
        self.compiled_graph = None
        self.checkpointer = MemorySaver()

        # Active sessions
        self.active_sessions: Dict[str, InterviewContext] = {}
        
    async def initialize(self):
        """Initialize all components"""
        try:
            await self._setup_llm_and_memory()
            await self._setup_database()
            await self._setup_message_bus()
            await self._setup_langgraph()

            logger.info("✅ Coordinator Agent initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinator: {e}")
            raise
    
    async def _setup_llm_and_memory(self):
        """Setup LangChain LLM and memory"""
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here":
            raise ValueError("CLAUDE_API_KEY not set")
            
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=2000
        )
        
        # Memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=20,  # Keep last 20 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        logger.info("✅ LangChain LLM and memory initialized")
    
    async def _setup_database(self):
        """Setup database connection for problem management"""
        try:
            database_url = os.getenv("DATABASE_URL")
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10
            )
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    async def _setup_message_bus(self):
        """Setup message bus connections"""
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        self.message_bus = await get_message_bus(redis_url)
        
        # Register agent
        await self.message_bus.register_agent(self.agent_name, {
            "type": "coordinator",
            "capabilities": ["orchestration", "routing", "problem_selection", "interview_management"],
            "status": "initializing"
        })
        
        # Subscribe to relevant channels
        self.message_bus.subscribe(Channels.COORDINATION, self.handle_coordination_message)
        self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_user_interaction)
        self.message_bus.subscribe(Channels.CODE_ANALYSIS, self.handle_code_analysis)
        self.message_bus.subscribe(Channels.EXECUTION, self.handle_execution_result)
        self.message_bus.subscribe(Channels.SYSTEM, self.handle_system_message)
        
        logger.info("✅ Message bus subscriptions established")
    
    async def _setup_langgraph(self):
        """Setup LangGraph workflow for interview coordination"""

        # Define the state graph
        workflow = StateGraph(InterviewWorkflowState)

        # Add nodes (functions that process state)
        workflow.add_node("analyze_input", self._analyze_input_node)
        workflow.add_node("route_decision", self._route_decision_node)
        workflow.add_node("select_problem", self._select_problem_node)
        workflow.add_node("handle_user_message", self._handle_user_message_node)
        workflow.add_node("process_code_submission", self._process_code_submission_node)
        workflow.add_node("process_execution", self._process_execution_node)

        # Set entry point
        workflow.set_entry_point("analyze_input")

        # Add conditional edges based on state
        workflow.add_conditional_edges(
            "analyze_input",
            self._route_based_on_action,
            {
                "start_interview": "select_problem",
                "user_message": "handle_user_message",
                "code_submission": "process_code_submission",
                "execute_code": "process_execution",
                "end": END
            }
        )

        # Connect remaining nodes to route_decision
        workflow.add_edge("select_problem", "route_decision")
        workflow.add_edge("handle_user_message", "route_decision")
        workflow.add_edge("process_code_submission", "route_decision")
        workflow.add_edge("process_execution", "route_decision")
        workflow.add_edge("route_decision", END)

        # Compile the graph with checkpointer for state persistence
        self.workflow = workflow
        self.compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("✅ LangGraph workflow initialized")

    # LangGraph Node Functions
    def _analyze_input_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Analyze input and prepare for routing"""
        logger.info(f"🔍 Analyzing input: action={state.get('current_action')}")

        # Add system message about coordinator role
        system_msg = SystemMessage(content="""You are an experienced technical interview coordinator.
Your role is to orchestrate the interview flow, select appropriate problems, and route messages to specialized agents.""")

        if "messages" not in state or not state["messages"]:
            state["messages"] = [system_msg]
        elif not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
            state["messages"].insert(0, system_msg)

        return state

    def _route_based_on_action(self, state: InterviewWorkflowState) -> str:
        """Conditional routing based on current action"""
        action = state.get("current_action", "")
        logger.info(f"🔀 Routing based on action: {action}")

        if action == "start_interview":
            return "start_interview"
        elif action == "user_message":
            return "user_message"
        elif action == "submit_code":
            return "code_submission"
        elif action == "execute_code":
            return "execute_code"
        else:
            return "end"

    def _select_problem_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Select appropriate problem for the interview"""
        logger.info("🎯 Selecting problem...")

        # This would normally be async, but nodes need to be sync
        # We'll handle async operations through the message handler
        state["next_step"] = "problem_selected"
        state["routing_decision"] = "interviewer"
        return state

    def _handle_user_message_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Handle user message and determine routing"""
        user_input = state.get("user_input", "")
        logger.info(f"💬 Processing user message: {user_input[:50]}...")

        # Use LLM to determine routing
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=f"User says: {user_input}\nWhich agent should handle this? (interviewer/code_analyzer/hint_provider)"))

        # Simple keyword-based routing for now
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ["hint", "help", "stuck"]):
            state["routing_decision"] = "hint_provider"
        elif any(word in user_input_lower for word in ["code", "solution"]):
            state["routing_decision"] = "code_analyzer"
        else:
            state["routing_decision"] = "interviewer"

        state["messages"] = messages
        return state

    def _process_code_submission_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Process code submission"""
        logger.info("📝 Processing code submission...")
        state["routing_decision"] = "code_analyzer"
        state["next_step"] = "code_analysis"
        return state

    def _process_execution_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Process code execution request"""
        logger.info("⚡ Processing execution request...")
        state["routing_decision"] = "execution"
        state["next_step"] = "execute"
        return state

    def _route_decision_node(self, state: InterviewWorkflowState) -> InterviewWorkflowState:
        """Final routing decision and message publication"""
        routing = state.get("routing_decision", "interviewer")
        logger.info(f"✅ Routing to: {routing}")
        state["next_step"] = "complete"
        return state

    # Helper methods for coordination logic
    def select_problem_tool(self, difficulty: str = "medium") -> str:
        """Tool: Select appropriate problem"""
        try:
            # This would normally be async, but CrewAI tools need to be sync
            # We'll handle this with a wrapper
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to handle differently
                future = asyncio.ensure_future(self._async_select_problem(difficulty))
                # For now, return a placeholder - we'll improve this
                return f"Problem selection initiated for difficulty: {difficulty}"
            else:
                return asyncio.run(self._async_select_problem(difficulty))
        except Exception as e:
            return f"Error selecting problem: {e}"
    
    async def _async_select_problem(self, difficulty: str) -> str:
        """Async helper for problem selection"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT id, title, description FROM problems WHERE difficulty = $1 ORDER BY RANDOM() LIMIT 1",
                    difficulty
                )
                if result:
                    return f"Selected problem: {result['title']} (ID: {result['id']})"
                else:
                    return f"No problems found for difficulty: {difficulty}"
        except Exception as e:
            return f"Database error: {e}"
    
    def route_message_tool(self, user_input: str, context: str = "") -> str:
        """Tool: Determine which agent should handle user input"""
        try:
            # Simple routing logic for now - we'll make this more sophisticated
            user_input_lower = user_input.lower()
            
            if any(word in user_input_lower for word in ["hint", "help", "stuck", "confused"]):
                return "route_to:hint_provider"
            elif any(word in user_input_lower for word in ["run", "execute", "test", "output"]):
                return "route_to:execution"
            elif any(word in user_input_lower for word in ["code", "solution", "implementation"]):
                return "route_to:code_analyzer"
            elif any(word in user_input_lower for word in ["problem", "question", "clarify", "explain"]):
                return "route_to:interviewer"
            else:
                return "route_to:interviewer"  # Default to interviewer
        except Exception as e:
            return f"Error routing message: {e}"
    
    def update_interview_state_tool(self, session_id: str, new_state: str, context_updates: str = "") -> str:
        """Tool: Update interview state"""
        try:
            # Update the session context
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                if hasattr(InterviewState, new_state.upper()):
                    context.interview_state = InterviewState(new_state.lower())
                    return f"Interview state updated to: {new_state}"
                else:
                    return f"Invalid state: {new_state}"
            else:
                return f"Session not found: {session_id}"
        except Exception as e:
            return f"Error updating state: {e}"
    
    def analyze_user_progress_tool(self, session_id: str) -> str:
        """Tool: Analyze user's current progress"""
        try:
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                metrics = context.performance_metrics
                
                analysis = f"""User Progress Analysis:
                - Time elapsed: {metrics.current_duration_seconds}s
                - Code changes: {metrics.code_changes}
                - Execution attempts: {metrics.execution_attempts}
                - Hints used: {metrics.hints_used}
                - Questions asked: {metrics.questions_asked}
                - Current state: {context.interview_state.value}
                """
                return analysis
            else:
                return f"Session not found: {session_id}"
        except Exception as e:
            return f"Error analyzing progress: {e}"
    
    # Message handlers
    async def handle_coordination_message(self, message: AgentMessage):
        """Handle coordination messages (interview lifecycle)"""
        try:
            payload = message.payload
            action = payload.get("action", "")
            
            logger.info(f"🎯 Received coordination action: {action}")
            
            if action == "start_interview":
                await self._start_interview(payload)
            elif action == "end_interview":
                await self._end_interview(payload)
            elif action == "pause_interview":
                await self._pause_interview(payload)
            elif action == "submit_code":
                logger.info(f"📝 Processing submit_code action for session: {payload.get('session_id', 'unknown')}")
                await self._handle_code_submission(payload)
            elif action == "execute_code":
                logger.info(f"▶️ Processing execute_code action for session: {payload.get('session_id', 'unknown')}")
                await self._handle_execution_request(payload)
            else:
                logger.warning(f"Unknown coordination action: {action}")
                
        except Exception as e:
            logger.error(f"❌ Error handling coordination message: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    async def handle_user_interaction(self, message: AgentMessage):
        """Handle user input and route appropriately"""
        try:
            # Only handle messages targeting coordinator
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            session_id = payload.get("session_id", "")
            user_content = payload.get("content", "")
            
            if not session_id or not user_content:
                logger.warning("Missing session_id or content in user interaction")
                return
            
            # Get or create context
            context = await self._get_or_create_context(session_id, payload.get("user_id", "unknown"))
            
            # Update conversation history
            context.add_conversation("user", user_content)
            await self.message_bus.store_context(session_id, context)

            # Use LangGraph to decide how to handle this input
            await self._process_user_input_with_langgraph(session_id, user_content, context)
            
        except Exception as e:
            logger.error(f"❌ Error handling user interaction: {e}")
    
    async def handle_code_analysis(self, message: AgentMessage):
        """Handle code analysis results"""
        try:
            payload = message.payload
            session_id = payload.get("session_id", "")
            
            if session_id in self.active_sessions:
                # Update context with code analysis results
                context = self.active_sessions[session_id]
                # Process code analysis and potentially trigger next steps
                await self._process_code_analysis(session_id, payload)
                
        except Exception as e:
            logger.error(f"❌ Error handling code analysis: {e}")
    
    async def handle_execution_result(self, message: AgentMessage):
        """Handle code execution results"""
        try:
            payload = message.payload
            session_id = payload.get("session_id", "")
            
            if session_id in self.active_sessions:
                await self._process_execution_result(session_id, payload)
                
        except Exception as e:
            logger.error(f"❌ Error handling execution result: {e}")
    
    async def handle_system_message(self, message: AgentMessage):
        """Handle system-level messages"""
        try:
            payload = message.payload
            event_type = payload.get("event", "")
            
            if event_type == "health_check":
                await self.message_bus.heartbeat(self.agent_name)
            elif event_type == "shutdown":
                await self.shutdown()
                
        except Exception as e:
            logger.error(f"❌ Error handling system message: {e}")
    
    # Core interview orchestration methods
    async def _start_interview(self, payload: Dict[str, Any]):
        """Start a new interview session"""
        try:
            session_id = payload.get("session_id", str(uuid.uuid4()))
            user_id = payload.get("user_id", "anonymous")
            difficulty = payload.get("difficulty", "medium")
            
            # Create interview context
            context = InterviewContext(
                session_id=session_id,
                user_id=user_id,
                interview_state=InterviewState.PROBLEM_INTRODUCTION
            )
            
            # Store context
            self.active_sessions[session_id] = context
            await self.message_bus.store_context(session_id, context)

            # Use LangGraph workflow to select appropriate problem and start interview
            initial_state = InterviewWorkflowState(
                session_id=session_id,
                user_id=user_id,
                current_action="start_interview",
                user_input=None,
                interview_state=InterviewState.PROBLEM_INTRODUCTION.value,
                current_problem=None,
                routing_decision=None,
                messages=[],
                context={"difficulty": difficulty},
                next_step=None
            )

            # Execute workflow
            config = {"configurable": {"thread_id": session_id}}
            result_state = self.compiled_graph.invoke(initial_state, config)

            # Select and load the problem
            problem = await self._select_problem(difficulty)
            if problem:
                context.current_problem = problem
                context.interview_state = InterviewState.PROBLEM_INTRODUCTION
                self.active_sessions[session_id] = context
                await self.message_bus.store_context(session_id, context)
            
            # Send introduction to interviewer agent
            await self._route_to_interviewer(session_id, "start_interview", {
                "problem": problem.to_dict() if problem else None,
                "difficulty": difficulty,
                "user_id": user_id
            })
            
            logger.info(f"✅ Interview started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting interview: {e}")
    
    async def _process_user_input_with_langgraph(self, session_id: str, user_input: str, context: InterviewContext):
        """Process user input using LangGraph workflow"""
        try:
            # Create initial state for the workflow
            initial_state = InterviewWorkflowState(
                session_id=session_id,
                user_id=context.user_id,
                current_action="user_message",
                user_input=user_input,
                interview_state=context.interview_state.value,
                current_problem=context.current_problem.to_dict() if context.current_problem else None,
                routing_decision=None,
                messages=[],
                context={},
                next_step=None
            )

            # Execute the workflow
            config = {"configurable": {"thread_id": session_id}}
            result_state = self.compiled_graph.invoke(initial_state, config)

            # Execute the routing decision
            routing_decision = result_state.get("routing_decision", "interviewer")
            await self._execute_routing_decision(session_id, user_input, f"route_to:{routing_decision}", context)

        except Exception as e:
            logger.error(f"❌ Error processing user input with LangGraph: {e}")
            # Fallback to simple routing
            await self._route_to_interviewer(session_id, "handle_user_input", {"content": user_input})
    
    async def _execute_routing_decision(self, session_id: str, user_input: str, decision: str, context: InterviewContext):
        """Execute the routing decision made by LangGraph"""
        try:
            # For now, simple parsing - we'll improve this
            if "interviewer" in decision.lower():
                await self._route_to_interviewer(session_id, "handle_user_input", {"content": user_input})
            elif "code_analyzer" in decision.lower():
                await self._route_to_code_analyzer(session_id, "analyze_input", {"content": user_input})
            elif "hint" in decision.lower():
                await self._route_to_hint_provider(session_id, "provide_hint", {"content": user_input})
            else:
                # Default to interviewer
                await self._route_to_interviewer(session_id, "handle_user_input", {"content": user_input})
                
        except Exception as e:
            logger.error(f"❌ Error executing routing decision: {e}")
    
    # Agent routing methods
    async def _route_to_interviewer(self, session_id: str, action: str, payload: Dict[str, Any]):
        """Route message to interviewer agent"""
        message = AgentMessage(
            event_type=EventType.AGENT_RESPONSE,
            source_agent=self.agent_name,
            target_agent="interviewer",
            payload={
                "action": action,
                "session_id": session_id,
                **payload
            },
            context_snapshot=self.active_sessions.get(session_id, {}).to_dict() if session_id in self.active_sessions else {}
        )
        await self.message_bus.publish(Channels.USER_INTERACTION, message)
    
    async def _route_to_code_analyzer(self, session_id: str, action: str, payload: Dict[str, Any]):
        """Route message to code analyzer agent"""
        message = AgentMessage(
            event_type=EventType.CODE_CHANGE,
            source_agent=self.agent_name,
            target_agent="code_analyzer",
            payload={
                "action": action,
                "session_id": session_id,
                **payload
            },
            context_snapshot=self.active_sessions.get(session_id, {}).to_dict() if session_id in self.active_sessions else {}
        )
        await self.message_bus.publish(Channels.CODE_ANALYSIS, message)
    
    async def _route_to_hint_provider(self, session_id: str, action: str, payload: Dict[str, Any]):
        """Route message to hint provider agent"""
        message = AgentMessage(
            event_type=EventType.HINT_REQUEST,
            source_agent=self.agent_name,
            target_agent="hint_provider",
            payload={
                "action": action,
                "session_id": session_id,
                **payload
            },
            context_snapshot=self.active_sessions.get(session_id, {}).to_dict() if session_id in self.active_sessions else {}
        )
        await self.message_bus.publish(Channels.USER_INTERACTION, message)
    
    async def _route_to_execution_agent(self, session_id: str, action: str, payload: Dict[str, Any]):
        """Route message to execution agent"""
        message = AgentMessage(
            event_type=EventType.EXECUTION_REQUEST,
            source_agent=self.agent_name,
            target_agent="execution",
            payload={
                "action": action,
                "session_id": session_id,
                **payload
            },
            context_snapshot=self.active_sessions.get(session_id, {}).to_dict() if session_id in self.active_sessions else {}
        )
        await self.message_bus.publish(Channels.EXECUTION, message)
    
    async def _smart_route_user_input(self, user_input: str, context: InterviewContext) -> str:
        """Smart routing based on user input content and context"""
        try:
            user_input_lower = user_input.lower()
            
            # Route based on input patterns
            hint_keywords = ["hint", "help", "stuck", "clue", "guidance", "approach"]
            if any(keyword in user_input_lower for keyword in hint_keywords):
                return "route_to:hint_provider"
            
            # Questions about the problem or clarifications
            question_keywords = ["what", "how", "why", "can i", "should i", "example", "clarify"]
            if any(keyword in user_input_lower for keyword in question_keywords):
                return "route_to:interviewer"
            
            # General conversation
            conversation_keywords = ["hi", "hello", "thanks", "ok", "yes", "no", "ready", "start"]
            if any(keyword in user_input_lower for keyword in conversation_keywords):
                return "route_to:interviewer"
            
            # Default to interviewer for any other input
            return "route_to:interviewer"
            
        except Exception as e:
            logger.error(f"Error in smart routing: {e}")
            return "route_to:interviewer"
    
    # Helper methods
    async def _select_problem(self, difficulty: str) -> Optional[Problem]:
        """Select a random problem of given difficulty"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM problems WHERE difficulty = $1 ORDER BY RANDOM() LIMIT 1",
                    difficulty
                )
                
                if row:
                    return Problem(
                        id=row['id'],
                        title=row['title'],
                        description=row['description'],
                        difficulty=Difficulty(row['difficulty']),
                        test_cases=[],  # We'll load these separately
                        starter_code=row['starter_code'] or "",
                        hints=row['hints'] or [],
                        topics=row['topics'] or []
                    )
                return None
                
        except Exception as e:
            logger.error(f"❌ Error selecting problem: {e}")
            return None
    
    async def _get_or_create_context(self, session_id: str, user_id: str) -> InterviewContext:
        """Get existing context or create new one"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from Redis
        context_data = await self.message_bus.get_context(session_id)
        if context_data:
            # Reconstruct context from stored data
            context = InterviewContext(
                session_id=session_id,
                user_id=user_id,
                interview_state=InterviewState(context_data.get("interview_state", "initializing"))
            )
            self.active_sessions[session_id] = context
            return context
        
        # Create new context
        context = InterviewContext(session_id=session_id, user_id=user_id)
        self.active_sessions[session_id] = context
        return context
    
    async def _process_code_analysis(self, session_id: str, payload: Dict[str, Any]):
        """Process code analysis results and decide next steps"""
        # Implementation for handling code analysis results
        pass
    
    async def _process_execution_result(self, session_id: str, payload: Dict[str, Any]):
        """Process execution results and provide feedback"""
        # Implementation for handling execution results
        pass
    
    async def _end_interview(self, payload: Dict[str, Any]):
        """End interview session"""
        session_id = payload.get("session_id", "")
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            context.interview_state = InterviewState.COMPLETED
            
            # Clean up
            del self.active_sessions[session_id]
            
            logger.info(f"✅ Interview ended for session {session_id}")
    
    async def _pause_interview(self, payload: Dict[str, Any]):
        """Pause interview session"""
        # Implementation for pausing interviews
        pass
    
    async def _handle_code_submission(self, payload: Dict[str, Any]):
        """Handle code submission and route to code analyzer"""
        try:
            session_id = payload.get("session_id", "")
            code = payload.get("code", "")
            language = payload.get("language", "python")
            
            logger.info(f"🔍 Routing code submission to analyzer for session {session_id}")
            
            # Route to code analyzer agent
            await self._route_to_code_analyzer(session_id, "analyze_code", {
                "code": code,
                "language": language,
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling code submission: {e}")
    
    async def _handle_execution_request(self, payload: Dict[str, Any]):
        """Handle code execution request and route to execution agent"""
        try:
            session_id = payload.get("session_id", "")
            
            logger.info(f"⚡ Routing execution request to execution agent for session {session_id}")
            
            # Route to execution agent
            await self._route_to_execution_agent(session_id, "execute_code", {
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling execution request: {e}")
    
    # Main run loop
    async def run(self):
        """Main coordinator run loop"""
        self.running = True
        logger.info("🚀 Coordinator Agent starting...")
        
        try:
            await self.initialize()
            
            # Update status to active
            await self.message_bus.register_agent(self.agent_name, {
                "type": "coordinator",
                "capabilities": ["orchestration", "routing", "problem_selection", "interview_management"],
                "status": "active"
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start listening for messages
            await self.message_bus.start_listening()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in coordinator: {e}")
        finally:
            await self.shutdown()
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                await self.message_bus.heartbeat(self.agent_name)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                break
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Coordinator shutting down...")
        self.running = False
        
        if self.message_bus:
            await self.message_bus.shutdown()
        if self.db_pool:
            await self.db_pool.close()

# Entry point
async def main():
    coordinator = CoordinatorAgent()
    try:
        await coordinator.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())