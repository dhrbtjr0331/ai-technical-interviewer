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
from langchain.tools import Tool
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
            k=15,  # Keep last 15 exchanges
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
            # Conversation chain for general chat
            conversation_prompt = PromptTemplate(
                input_variables=["user_input", "interview_context", "personality_mode", "chat_history"],
                template="""You are an experienced technical interviewer at a top tech company. 

PERSONALITY MODE: {personality_mode}
- encouraging: Be warm, supportive, help build confidence
- challenging: Be more direct, push for deeper thinking
- neutral: Balanced professional approach
- clarifying: Focus on explaining and helping understanding

INTERVIEW CONTEXT:
{interview_context}

CONVERSATION HISTORY:
{chat_history}

USER INPUT: "{user_input}"

Respond naturally as an interviewer would. Keep responses conversational and appropriately sized (2-4 sentences typically). 
Match the personality mode while maintaining professionalism. Don't be overly verbose.

Response:"""
            )
            
            self.conversation_chain = LLMChain(
                llm=self.llm,
                prompt=conversation_prompt,
                memory=self.memory
            )
            
            # Problem introduction chain
            problem_intro_prompt = PromptTemplate(
                input_variables=["problem_title", "problem_description", "difficulty", "user_context"],
                template="""You are introducing a coding problem to a candidate in a technical interview.

PROBLEM: {problem_title} ({difficulty} difficulty)
DESCRIPTION: {problem_description}

USER CONTEXT: {user_context}

Create a warm, professional introduction that:
1. Welcomes them to the interview segment
2. Introduces the problem clearly 
3. Explains what they need to do
4. Encourages them to ask questions if anything is unclear
5. Sets a collaborative, supportive tone

Keep it conversational and encouraging. This should feel like a real interview, not a robot.

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
            # Define tools for the interviewer
            tools = [
                Tool(
                    name="analyze_user_emotional_state",
                    description="Analyze user's emotional state and confidence level from their recent interactions",
                    func=self._tool_analyze_emotional_state
                ),
                Tool(
                    name="get_problem_guidance",
                    description="Get appropriate guidance or hints for the current problem without giving away the solution",
                    func=self._tool_get_problem_guidance
                ),
                Tool(
                    name="assess_progress",
                    description="Assess user's progress and determine what kind of response would be most helpful",
                    func=self._tool_assess_progress
                ),
                Tool(
                    name="determine_personality_mode",
                    description="Determine what personality mode to use based on user state and context",
                    func=self._tool_determine_personality_mode
                )
            ]
            
            # Create the interviewer CrewAI agent
            self.crew_agent = Agent(
                role="Technical Interview Specialist",
                goal="Conduct natural, effective technical interviews that assess skills while maintaining positive candidate experience",
                backstory="""You are a senior software engineer with 8+ years of experience conducting 
                technical interviews at companies like Google, Meta, and Amazon. You have interviewed 
                over 500 candidates and are known for your ability to:
                
                - Make candidates feel comfortable while still assessing thoroughly
                - Adapt your communication style to different personality types
                - Guide candidates when they're stuck without giving away answers
                - Ask clarifying questions that reveal thought processes
                - Provide encouragement that builds confidence
                - Recognize when to push harder vs. when to support
                
                You understand that interviews are stressful and your job is to help candidates 
                show their best work while getting an accurate assessment of their abilities.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=tools,
                memory=True
            )
            
            # Create the crew
            self.crew = Crew(
                agents=[self.crew_agent],
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
    
    def _tool_analyze_emotional_state(self, recent_interactions: str) -> str:
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
    
    def _tool_get_problem_guidance(self, problem_id: str, user_approach: str = "") -> str:
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
    
    def _tool_assess_progress(self, session_context: str) -> str:
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
    
    def _tool_determine_personality_mode(self, emotional_state: str, progress_assessment: str, time_context: str) -> str:
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
                    problem_title=problem_title,
                    problem_description=problem_description,
                    difficulty=difficulty,
                    user_context=user_context
                )
            
            # Store session context
            self.active_sessions[session_id] = {
                "current_problem": problem_data,
                "difficulty": difficulty,
                "start_time": datetime.now(),
                "conversation_history": []
            }
            
            # Send response to user
            await self._send_response_to_user(session_id, response, "interview_start")
            
            logger.info(f"✅ Interview started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling interview start: {e}")
            # Fallback response
            await self._send_response_to_user(
                session_id, 
                "Welcome to your technical interview! I'm here to guide you through a coding problem. Let's begin - feel free to ask any questions as we go.",
                "interview_start_fallback"
            )
    
    async def _handle_interview_start_fallback(self, session_id: str, payload: Dict[str, Any]):
        """Handle fallback interview start when coordinator has issues"""
        try:
            difficulty = payload.get("difficulty", "medium")
            error = payload.get("error", "")
            
            response = f"""Welcome to your technical interview! I'm your interviewer today.
            
We'll be working on a {difficulty} level coding problem. I'm here to help guide you through it and answer any questions you might have.

Let me get the problem details ready for you. In the meantime, feel free to let me know if you have any questions about the interview process or if there's anything I can clarify.

Ready to get started?"""
            
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
                
                task = Task(
                    description=f"""
                    The candidate is asking for clarification about the problem: "{user_content}"
                    
                    Current problem: {problem_data.get('title', 'Unknown')}
                    Problem description: {problem_data.get('description', 'No description available')}
                    
                    Use your tools to:
                    1. Assess what specific aspect they're confused about
                    2. Determine the best way to clarify without giving away the solution
                    3. Provide a helpful clarification that guides them forward
                    
                    Respond as a helpful interviewer who wants them to succeed.
                    """,
                    agent=self.crew_agent,
                    expected_output="Clear, helpful clarification response"
                )
                
                response = str(self.crew.kickoff(tasks=[task]))
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
            
            task = Task(
                description=f"""
                Respond to this candidate input during the interview: "{user_content}"
                
                Recent conversation context: {recent_interactions}
                Session context: {context_json}
                
                Use your tools to:
                1. Analyze their emotional state from recent interactions
                2. Assess their progress and needs
                3. Determine the best personality mode for your response
                
                Then provide a natural, helpful response that matches the determined personality mode.
                Be conversational and supportive while maintaining interview professionalism.
                """,
                agent=self.crew_agent,
                expected_output="Natural, contextually appropriate interviewer response"
            )
            
            response = str(self.crew.kickoff(tasks=[task]))
            
            # Add interviewer response to history
            conversation_history.append({"speaker": "interviewer", "content": response, "timestamp": datetime.now()})
            
            # Update session data
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["conversation_history"] = conversation_history
            
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
            heartbeat