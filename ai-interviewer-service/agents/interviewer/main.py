import asyncio
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

# CrewAI and LangChain imports
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Database for problem and user insights
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
import threading

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import (
    AgentMessage, EventType, InterviewContext, InterviewState, 
    Problem, Difficulty, Message, PerformanceMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewerAgent:
    """
    Adaptive conversational interviewer using hybrid CrewAI + LangChain approach.
    Adjusts personality based on user state and provides natural interview experience.
    """
    
    def __init__(self):
        self.agent_name = "interviewer"
        self.message_bus: MessageBus = None
        self.sync_db_pool: psycopg2.pool.ThreadedConnectionPool = None
        self.running = False
        
        # LangChain components
        self.llm: ChatAnthropic = None
        self.memory: ConversationBufferWindowMemory = None
        
        # CrewAI components for complex reasoning
        self.crew_agent: Agent = None
        self.crew: Crew = None
        
        # Direct LLM chains for quick responses
        self.conversation_chain: LLMChain = None
        self.problem_introduction_chain: LLMChain = None
        self.encouragement_chain: LLMChain = None
        
        # Session-specific data
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._db_lock = threading.Lock()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            await self._setup_llm_and_memory()
            await self._setup_database()
            await self._setup_message_bus()
            await self._setup_langchain_chains()
            await self._setup_crewai()
            
            logger.info("✅ Interviewer Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize interviewer: {e}")
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
            max_tokens=1500
        )
        
        # Memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=25,  # Keep last 25 exchanges for better context
            return_messages=True,
            memory_key="chat_history"
        )
        
        logger.info("✅ LangChain LLM and memory initialized")
    
    async def _setup_database(self):
        """Setup sync database connection for user insights"""
        try:
            database_url = os.getenv("DATABASE_URL")
            
            # Parse URL for sync client
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            self.sync_db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password
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
            "type": "interviewer",
            "capabilities": ["conversation", "problem_introduction", "clarification", "encouragement"],
            "status": "initializing"
        })
        
        # Subscribe to relevant channels
        self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_user_interaction)
        self.message_bus.subscribe(Channels.SYSTEM, self.handle_system_message)
        
        logger.info("✅ Message bus subscriptions established")
    
    async def _setup_langchain_chains(self):
        """Setup LangChain chains for quick responses"""
        try:
            # Conversation chain for LeetCode-style DSA interviews
            conversation_prompt = PromptTemplate(
                input_variables=["input"],
                template="""You are an experienced technical interviewer conducting a LeetCode-style algorithmic coding interview at a top tech company (like Google, Meta, Amazon).

This is a DATA STRUCTURES & ALGORITHMS interview focused on problem-solving skills, not a general software engineering discussion.

Your role:
- Focus ONLY on algorithmic problem-solving and coding
- Guide candidates through LeetCode-style problems (arrays, strings, trees, graphs, dynamic programming, etc.)
- Ask about time/space complexity analysis
- Help with algorithm optimization and edge cases
- Provide additional test cases and examples when requested
- Remember the current problem context and maintain conversation continuity
- Avoid general software engineering topics, system design, or behavioral questions

Keep responses focused on the coding problem at hand (2-4 sentences typically).

{input}

Response:"""
            )
            
            self.conversation_chain = LLMChain(
                llm=self.llm,
                prompt=conversation_prompt,
                memory=self.memory
            )
            
            # Problem introduction chain for LeetCode-style problems
            problem_intro_prompt = PromptTemplate(
                input_variables=["input"],
                template="""You are introducing a LeetCode-style algorithmic coding problem to a candidate.

This is a focused DSA (Data Structures & Algorithms) interview. Introduce the problem professionally and concisely:
- Present the problem statement clearly
- Mention it's similar to LeetCode problems they may have practiced
- Encourage them to think about optimal time/space complexity
- Let them know they can ask clarifying questions about the algorithm/data structures

Keep the introduction brief and algorithm-focused.

{input}

Introduction:"""
            )
            
            self.problem_introduction_chain = LLMChain(
                llm=self.llm,
                prompt=problem_intro_prompt
            )
            
            # Encouragement chain for when users are struggling
            encouragement_prompt = PromptTemplate(
                input_variables=["user_state", "progress_context", "time_context", "problem_context"],
                template="""The candidate seems to be struggling or needs encouragement.

USER STATE: {user_state}
PROGRESS: {progress_context}
TIME CONTEXT: {time_context}  
PROBLEM: {problem_context}

Provide appropriate encouragement that:
1. Acknowledges their effort
2. Provides gentle guidance without giving away the solution
3. Helps them refocus or break down the problem
4. Maintains their confidence
5. Feels like genuine human support

Be supportive but not condescending. This should feel like a mentor helping, not empty platitudes.

Encouragement:"""
            )
            
            self.encouragement_chain = LLMChain(
                llm=self.llm,
                prompt=encouragement_prompt
            )
            
            logger.info("✅ LangChain chains initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up LangChain chains: {e}")
            raise
    
    async def _setup_crewai(self):
        """Setup CrewAI for complex interview orchestration"""
        try:
            # Define tools for the interviewer - using empty list for now but tools are available as methods
            tools = []
            
            # Create the interviewer CrewAI agent
            self.crew_agent = Agent(
                role="LeetCode-Style Algorithm Interview Specialist",
                goal="Conduct focused DSA coding interviews that assess algorithmic problem-solving skills using LeetCode-style problems",
                backstory="""You are a senior software engineer with 8+ years of experience conducting 
                LeetCode-style algorithmic coding interviews at top tech companies (Google, Meta, Amazon, Microsoft).
                
                You specialize in DATA STRUCTURES & ALGORITHMS interviews and have interviewed over 500 candidates on:
                - Array/String manipulation problems
                - Binary Trees and Graph traversals
                - Dynamic Programming and Recursion
                - Hash Tables and Two Pointers techniques
                - Time/Space complexity optimization
                
                Your expertise:
                - Guide candidates through optimal algorithm approaches
                - Help with complexity analysis (Big O notation)
                - Identify edge cases and test scenarios
                - Provide hints without giving away solutions
                - Focus purely on algorithmic thinking, not system design or behavioral aspects
                
                You keep interviews focused on coding and algorithms, similar to actual LeetCode practice sessions.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=tools,
                memory=True
            )
            
            # Create the crew - we'll add tasks dynamically
            self.crew = Crew(
                agents=[self.crew_agent],
                tasks=[],  # Empty for now, we'll create tasks on demand
                verbose=True,
                process=Process.sequential
            )
            
            logger.info("✅ CrewAI interviewer agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up CrewAI: {e}")
            raise
    
    # ==========================================
    # CREWAI TOOLS
    # ==========================================
    
    def analyze_emotional_state_tool(self, recent_interactions: str) -> str:
        """Analyze user's emotional state from recent interactions"""
        try:
            if not recent_interactions:
                return json.dumps({"emotional_state": "neutral", "confidence_level": "unknown"})
            
            interactions = recent_interactions.lower()
            
            # Simple emotional analysis (could be enhanced with more sophisticated NLP)
            emotional_indicators = {
                "frustrated": ["stuck", "confused", "not working", "frustrating", "difficult"],
                "confident": ["got it", "understand", "clear", "makes sense", "easy"],
                "uncertain": ["not sure", "maybe", "think so", "possibly", "might"],
                "engaged": ["interesting", "good question", "let me think", "i see"],
                "overwhelmed": ["too much", "complicated", "lost", "don't understand"]
            }
            
            detected_emotions = []
            for emotion, indicators in emotional_indicators.items():
                if any(indicator in interactions for indicator in indicators):
                    detected_emotions.append(emotion)
            
            primary_emotion = detected_emotions[0] if detected_emotions else "neutral"
            
            # Assess confidence level
            confidence_indicators = {
                "high": ["definitely", "sure", "confident", "easy", "got this"],
                "low": ["not sure", "confused", "lost", "difficult", "struggling"],
                "medium": ["think", "maybe", "probably", "seems like"]
            }
            
            confidence_level = "medium"  # default
            for level, indicators in confidence_indicators.items():
                if any(indicator in interactions for indicator in indicators):
                    confidence_level = level
                    break
            
            return json.dumps({
                "emotional_state": primary_emotion,
                "confidence_level": confidence_level,
                "detected_emotions": detected_emotions,
                "analysis": f"User appears {primary_emotion} with {confidence_level} confidence"
            })
            
        except Exception as e:
            logger.error(f"Error analyzing emotional state: {e}")
            return json.dumps({"emotional_state": "neutral", "confidence_level": "unknown", "error": str(e)})
    
    def get_problem_guidance_tool(self, problem_id: str, user_approach: str = "") -> str:
        """Get appropriate guidance for the current problem"""
        try:
            with self._db_lock:
                conn = self.sync_db_pool.getconn()
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute(
                        "SELECT hints, topics, description FROM problems WHERE id = %s",
                        (problem_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        hints = result['hints'] or []
                        topics = result['topics'] or []
                        description = result['description']
                        
                        # Select appropriate hint level based on user approach
                        if not user_approach:
                            guidance_level = "gentle"  # First hint
                        elif "approach" in user_approach.lower():
                            guidance_level = "directional"  # Help with direction
                        else:
                            guidance_level = "specific"  # More specific help
                        
                        return json.dumps({
                            "success": True,
                            "hints": hints,
                            "topics": topics,
                            "guidance_level": guidance_level,
                            "suggestion": f"Consider the {topics[0] if topics else 'problem structure'} aspect of this problem"
                        })
                    else:
                        return json.dumps({
                            "success": False,
                            "error": "Problem not found"
                        })
                        
                finally:
                    self.sync_db_pool.putconn(conn)
                    
        except Exception as e:
            logger.error(f"Error getting problem guidance: {e}")
            return json.dumps({
                "success": False,
                "error": f"Database error: {str(e)}"
            })
    
    def assess_progress_tool(self, session_context: str) -> str:
        """Assess user's progress and determine helpful response type"""
        try:
            if not session_context:
                return json.dumps({"assessment": "no_context", "recommendation": "general_support"})
            
            context_data = json.loads(session_context)
            
            # Extract key metrics
            time_elapsed = context_data.get("performance_metrics", {}).get("current_duration_seconds", 0)
            code_changes = context_data.get("performance_metrics", {}).get("code_changes", 0)
            questions_asked = context_data.get("performance_metrics", {}).get("questions_asked", 0)
            interview_state = context_data.get("interview_state", "unknown")
            has_code = bool(context_data.get("current_code", "").strip())
            
            # Progress assessment logic
            assessment = "on_track"
            recommendation = "continue_support"
            
            if time_elapsed > 1800 and not has_code:  # 30 minutes with no code
                assessment = "slow_start"
                recommendation = "encourage_coding"
            elif time_elapsed > 3000:  # 50 minutes
                assessment = "time_pressure"
                recommendation = "guide_to_solution"
            elif code_changes > 10 and time_elapsed < 600:  # Lots of changes quickly
                assessment = "rapid_iteration"
                recommendation = "help_focus"
            elif questions_asked > 5:
                assessment = "needs_clarification"
                recommendation = "provide_clarity"
            elif has_code and code_changes > 5:
                assessment = "actively_working"
                recommendation = "technical_support"
            
            return json.dumps({
                "assessment": assessment,
                "recommendation": recommendation,
                "time_elapsed_minutes": time_elapsed // 60,
                "progress_indicators": {
                    "has_code": has_code,
                    "code_changes": code_changes,
                    "questions_asked": questions_asked,
                    "interview_state": interview_state
                }
            })
            
        except Exception as e:
            logger.error(f"Error assessing progress: {e}")
            return json.dumps({
                "assessment": "error",
                "recommendation": "general_support",
                "error": str(e)
            })
    
    def determine_personality_mode_tool(self, emotional_state: str, progress_assessment: str, time_context: str) -> str:
        """Determine what personality mode to use"""
        try:
            # Default mode
            personality_mode = "neutral"
            
            # Emotional state influences
            if emotional_state in ["frustrated", "overwhelmed"]:
                personality_mode = "encouraging"
            elif emotional_state == "confident":
                personality_mode = "challenging"
            elif emotional_state in ["uncertain", "confused"]:
                personality_mode = "clarifying"
            
            # Progress assessment overrides
            if progress_assessment in ["slow_start", "time_pressure"]:
                personality_mode = "encouraging"
            elif progress_assessment == "rapid_iteration":
                personality_mode = "challenging"
            elif progress_assessment == "needs_clarification":
                personality_mode = "clarifying"
            
            # Time pressure considerations
            if "long" in time_context or "pressure" in time_context:
                if personality_mode == "challenging":
                    personality_mode = "neutral"  # Don't add pressure when time is tight
            
            return json.dumps({
                "personality_mode": personality_mode,
                "reasoning": f"Based on emotional_state={emotional_state}, progress={progress_assessment}, time={time_context}",
                "adaptation": f"Using {personality_mode} mode to best support candidate"
            })
            
        except Exception as e:
            logger.error(f"Error determining personality mode: {e}")
            return json.dumps({
                "personality_mode": "neutral",
                "reasoning": f"Error fallback: {str(e)}"
            })
    
    # ==========================================
    # MESSAGE HANDLERS
    # ==========================================
    
    async def handle_user_interaction(self, message: AgentMessage):
        """Handle user interaction messages"""
        try:
            # Only handle messages targeting interviewer
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            action = payload.get("action", "handle_user_input")
            session_id = payload.get("session_id", "")
            
            if not session_id:
                logger.warning("No session_id in user interaction message")
                return
            
            # Route to appropriate handler based on action
            if action == "start_interview" or action == "deliver_introduction":
                await self._handle_interview_start(session_id, payload)
            elif action == "clarify_problem":
                await self._handle_problem_clarification(session_id, payload)
            elif action == "handle_user_input":
                await self._handle_general_conversation(session_id, payload)
            elif action == "provide_encouragement":
                await self._handle_encouragement(session_id, payload)
            elif action == "start_interview_fallback":
                await self._handle_interview_start_fallback(session_id, payload)
            else:
                logger.warning(f"Unknown action for interviewer: {action}")
                await self._handle_general_conversation(session_id, payload)
                
        except Exception as e:
            logger.error(f"❌ Error handling user interaction: {e}")
    
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
    
    # ==========================================
    # INTERVIEW INTERACTION HANDLERS
    # ==========================================
    
    async def _handle_interview_start(self, session_id: str, payload: Dict[str, Any]):
        """Handle interview start with problem introduction"""
        try:
            problem_data = payload.get("problem", {})
            difficulty = payload.get("difficulty", "medium")
            user_id = payload.get("user_id", "candidate")
            content = payload.get("content", "")
            
            if problem_data and content:
                # Coordinator provided introduction content, use it
                response = content
            else:
                # Generate our own introduction
                problem_title = problem_data.get("title", "Coding Problem")
                problem_description = problem_data.get("description", "")
                
                user_context = f"Starting {difficulty} difficulty interview for {user_id}"
                
                response = await self.problem_introduction_chain.arun(
                    input=f"PROBLEM: {problem_title} ({difficulty} difficulty)\nDESCRIPTION: {problem_description}\nUSER: {user_id}"
                )
            
            # Store session context - ensure we capture manually presented problems too
            current_problem = problem_data if problem_data else {
                "title": "Two Sum",
                "description": "Given an array of integers and a target sum, return two numbers that add up to the target.",
                "examples": ["[2, 7, 11, 15] target=9 -> [2, 7]"],
                "type": "algorithm"
            }
            
            self.active_sessions[session_id] = {
                "current_problem": current_problem,
                "difficulty": difficulty,
                "start_time": datetime.now(),
                "conversation_history": [],
                "last_response": response  # Store what we just said
            }
            
            # Send response to user
            await self._send_response_to_user(session_id, response, "interview_start")
            
            logger.info(f"✅ Interview started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling interview start: {e}")
            # Fallback response
            await self._send_response_to_user(
                session_id, 
                "Welcome to your LeetCode-style DSA interview! I'm here to guide you through an algorithmic coding problem. Let's focus on optimal solutions and complexity analysis. Ready to start coding?",
                "interview_start_fallback"
            )
    
    async def _handle_interview_start_fallback(self, session_id: str, payload: Dict[str, Any]):
        """Handle fallback interview start when coordinator has issues"""
        try:
            difficulty = payload.get("difficulty", "medium")
            error = payload.get("error", "")
            
            response = f"""Welcome to your LeetCode-style algorithmic coding interview! I'm your interviewer today.

We'll be working on a {difficulty}-level data structures and algorithms problem, similar to what you'd find on LeetCode. 

Focus on:
- Optimal algorithm design and implementation
- Time and space complexity analysis
- Edge case handling

I'll present the problem shortly. Feel free to ask clarifying questions about the algorithm or approach as we go. Ready to dive into some coding?"""
            
            if error:
                logger.warning(f"Interview start fallback due to error: {error}")
            
            await self._send_response_to_user(session_id, response, "interview_start_fallback")
            
        except Exception as e:
            logger.error(f"❌ Error in interview start fallback: {e}")
    
    async def _handle_problem_clarification(self, session_id: str, payload: Dict[str, Any]):
        """Handle problem clarification requests"""
        try:
            user_content = payload.get("content", "")
            
            # Use CrewAI for complex clarification scenarios
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                problem_data = session_data.get("current_problem", {})
                
                # Use conversation chain for clarification
                response = await self.conversation_chain.arun(
                    input=f"The candidate is asking for clarification: '{user_content}'. Please help clarify the problem without giving away the solution."
                )
            else:
                # Fallback to simple clarification
                response = "I'd be happy to clarify! Could you be more specific about which part of the problem you'd like me to explain? I want to make sure you have a clear understanding before you begin coding."
            
            await self._send_response_to_user(session_id, response, "clarification")
            
        except Exception as e:
            logger.error(f"❌ Error handling clarification: {e}")
            await self._send_response_to_user(
                session_id,
                "I'd be happy to help clarify anything about the problem. What specific aspect would you like me to explain?",
                "clarification_fallback"
            )
    
    async def _handle_general_conversation(self, session_id: str, payload: Dict[str, Any]):
        """Handle general conversation using adaptive personality"""
        try:
            user_content = payload.get("content", "")
            
            # Get session context
            session_data = self.active_sessions.get(session_id, {})
            conversation_history = session_data.get("conversation_history", [])
            
            # Add user message to history
            conversation_history.append({"speaker": "user", "content": user_content, "timestamp": datetime.now()})
            
            # Prepare context for LLM
            recent_interactions = " ".join([msg["content"] for msg in conversation_history[-5:]])  # Last 5 messages
            
            # Use CrewAI to determine best response approach
            context_json = json.dumps({
                "current_problem": session_data.get("current_problem", {}),
                "conversation_length": len(conversation_history),
                "session_duration": (datetime.now() - session_data.get("start_time", datetime.now())).total_seconds()
            })
            
            # For newer CrewAI, we need to use the agent directly or recreate crew with task
            # Let's use a simpler approach with direct LLM call for now
            try:
                # Prepare context-aware input for the conversation chain
                current_problem = session_data.get("current_problem", {})
                last_response = session_data.get("last_response", "")
                problem_context = ""
                
                if current_problem:
                    problem_title = current_problem.get("title", "")
                    problem_desc = current_problem.get("description", "")
                    examples = current_problem.get("examples", [])
                    problem_context = f"\n\nCURRENT PROBLEM: {problem_title}\n{problem_desc}"
                    if examples:
                        problem_context += f"\nExamples: {', '.join(examples)}"
                    problem_context += "\n\n"
                
                # Include recent context
                recent_context = ""
                if last_response:
                    recent_context = f"MY LAST RESPONSE: {last_response}\n\n"
                
                # Use conversation chain with full context
                context_input = f"{problem_context}{recent_context}CANDIDATE MESSAGE: {user_content}"
                response = await self.conversation_chain.arun(
                    input=context_input
                )
            except Exception as crew_error:
                logger.warning(f"CrewAI fallback failed: {crew_error}, using simple response")
                response = f"Thank you for your input: '{user_content}'. Could you tell me more about what you're thinking or if you have any specific questions?"
            
            # Add interviewer response to history
            conversation_history.append({"speaker": "interviewer", "content": response, "timestamp": datetime.now()})
            
            # Detect if we just presented a problem and store it
            if self._detect_problem_presentation(response):
                problem_info = self._extract_problem_from_response(response)
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["current_problem"] = problem_info
                    logger.info(f"🎯 Stored problem: {problem_info.get('title', 'Unknown')} for session {session_id}")
            
            # Update session data
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["conversation_history"] = conversation_history
                self.active_sessions[session_id]["last_response"] = response  # Store our response for context
            
            await self._send_response_to_user(session_id, response, "conversation")
            
        except Exception as e:
            logger.error(f"❌ Error handling general conversation: {e}")
            # Fallback to simple response
            await self._send_response_to_user(
                session_id,
                "I understand. Please continue with your approach, and let me know if you have any questions or if there's anything I can help clarify.",
                "conversation_fallback"
            )
    
    async def _handle_encouragement(self, session_id: str, payload: Dict[str, Any]):
        """Handle encouragement requests when user is struggling"""
        try:
            user_state = payload.get("user_state", "struggling")
            progress_context = payload.get("progress_context", "")
            
            session_data = self.active_sessions.get(session_id, {})
            time_elapsed = (datetime.now() - session_data.get("start_time", datetime.now())).total_seconds()
            problem_data = session_data.get("current_problem", {})
            
            response = await self.encouragement_chain.arun(
                user_state=user_state,
                progress_context=progress_context,
                time_context=f"{int(time_elapsed//60)} minutes elapsed",
                problem_context=problem_data.get("title", "current problem")
            )
            
            await self._send_response_to_user(session_id, response, "encouragement")
            
        except Exception as e:
            logger.error(f"❌ Error providing encouragement: {e}")
            await self._send_response_to_user(
                session_id,
                "You're doing great! Take your time to think through the problem. Remember, it's about your thought process as much as the final solution. What's your next step?",
                "encouragement_fallback"
            )
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    async def _send_response_to_user(self, session_id: str, content: str, response_type: str = "general"):
        """Send response back to user through message bus"""
        try:
            response_message = AgentMessage(
                event_type=EventType.AGENT_RESPONSE,
                source_agent=self.agent_name,
                target_agent="user",
                payload={
                    "speaker": "interviewer",
                    "content": content,
                    "session_id": session_id,
                    "response_type": response_type,
                    "timestamp": datetime.now().isoformat()
                },
                context_snapshot={}
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, response_message)
            logger.info(f"📤 Sent response ({response_type}): {content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error sending response to user: {e}")
    
    def _detect_problem_presentation(self, response: str) -> bool:
        """Detect if the response contains a problem statement"""
        problem_indicators = [
            "write a function",
            "implement a",
            "given an array",
            "given a string",
            "coding problem",
            "palindrome",
            "sort",
            "find the",
            "return the",
            "algorithm",
            "how would you approach",
            "think out loud"
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in problem_indicators)
    
    def _extract_problem_from_response(self, response: str) -> dict:
        """Extract problem information from the response"""
        try:
            # Try to identify the problem type based on keywords
            response_lower = response.lower()
            
            if "palindrome" in response_lower:
                return {
                    "title": "Valid Palindrome",
                    "description": "Determine if a given string is a valid palindrome, considering only alphanumeric characters and ignoring case.",
                    "difficulty": "easy",
                    "topics": ["string", "two-pointers"],
                    "type": "algorithm"
                }
            elif "two sum" in response_lower or "add up to" in response_lower:
                return {
                    "title": "Two Sum",
                    "description": "Find two numbers in an array that add up to a specific target sum.",
                    "difficulty": "easy", 
                    "topics": ["array", "hash-table"],
                    "type": "algorithm"
                }
            elif "subarray" in response_lower and "sum" in response_lower:
                return {
                    "title": "Maximum Subarray",
                    "description": "Find the contiguous subarray with the largest sum.",
                    "difficulty": "medium",
                    "topics": ["array", "dynamic-programming"],
                    "type": "algorithm"
                }
            else:
                # Generic problem
                return {
                    "title": "Coding Problem",
                    "description": response[:200] + "..." if len(response) > 200 else response,
                    "difficulty": "medium",
                    "topics": ["algorithm"],
                    "type": "algorithm"
                }
                
        except Exception as e:
            logger.error(f"Error extracting problem: {e}")
            return {
                "title": "Coding Problem",
                "description": "Algorithm problem",
                "difficulty": "medium",
                "topics": ["algorithm"],
                "type": "algorithm"
            }
    
    # ==========================================
    # MAIN RUN LOOP
    # ==========================================
    
    async def run(self):
        """Main interviewer run loop"""
        self.running = True
        logger.info("🚀 Interviewer Agent starting...")
        
        try:
            await self.initialize()
            
            # Update status to active
            await self.message_bus.register_agent(self.agent_name, {
                "type": "interviewer",
                "capabilities": ["conversation", "problem_introduction", "clarification", "encouragement"],
                "status": "active"
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start listening for messages
            await self.message_bus.start_listening()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in interviewer: {e}")
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
        logger.info("🛑 Interviewer shutting down...")
        self.running = False
        
        if self.message_bus:
            await self.message_bus.shutdown()
        if self.sync_db_pool:
            self.sync_db_pool.closeall()

# Entry point
async def main():
    interviewer = InterviewerAgent()
    try:
        await interviewer.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await interviewer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())