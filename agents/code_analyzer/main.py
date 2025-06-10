import asyncio
import os
import logging
import json
import ast
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import uuid

# CrewAI and LangChain imports
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Code analysis tools
import ast
import tokenize
import io
from collections import defaultdict

# Database for problem context
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
import threading

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import (
    AgentMessage, EventType, InterviewContext, InterviewState, 
    Problem, Difficulty, Message, PerformanceMetrics, CodeSnapshot
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAnalyzerAgent:
    """
    Technical code analysis agent using AST parsing, pattern recognition, and LLM insights.
    Provides real-time feedback on code quality, approach, and potential improvements.
    """
    
    def __init__(self):
        self.agent_name = "code_analyzer"
        self.message_bus: MessageBus = None
        self.sync_db_pool: psycopg2.pool.ThreadedConnectionPool = None
        self.running = False
        
        # LangChain components
        self.llm: ChatAnthropic = None
        self.memory: ConversationBufferWindowMemory = None
        
        # CrewAI components for complex analysis
        self.crew_agent: Agent = None
        self.crew: Crew = None
        
        # LangChain chains for specific analysis tasks
        self.code_review_chain: LLMChain = None
        self.approach_analysis_chain: LLMChain = None
        self.feedback_generation_chain: LLMChain = None
        
        # Code analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._db_lock = threading.Lock()
        
        # Pattern recognition data
        self.algorithm_patterns = self._load_algorithm_patterns()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            await self._setup_llm_and_memory()
            await self._setup_database()
            await self._setup_message_bus()
            await self._setup_langchain_chains()
            await self._setup_crewai()
            
            logger.info("✅ Code Analyzer Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize code analyzer: {e}")
            raise
    
    async def _setup_llm_and_memory(self):
        """Setup LangChain LLM and memory"""
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here":
            raise ValueError("CLAUDE_API_KEY not set")
            
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0.3,  # Lower temperature for more consistent technical analysis
            max_tokens=2000
        )
        
        # Memory for code analysis context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 code interactions
            return_messages=True,
            memory_key="analysis_history"
        )
        
        logger.info("✅ LangChain LLM and memory initialized")
    
    async def _setup_database(self):
        """Setup database connection for problem context"""
        try:
            database_url = os.getenv("DATABASE_URL")
            
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
            "type": "code_analyzer",
            "capabilities": ["ast_analysis", "pattern_recognition", "code_review", "approach_detection"],
            "status": "initializing"
        })
        
        # Subscribe to relevant channels
        self.message_bus.subscribe(Channels.CODE_ANALYSIS, self.handle_code_analysis)
        self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_user_interaction)
        self.message_bus.subscribe(Channels.SYSTEM, self.handle_system_message)
        
        logger.info("✅ Message bus subscriptions established")
    
    async def _setup_langchain_chains(self):
        """Setup LangChain chains for code analysis"""
        try:
            # Code review chain
            code_review_prompt = PromptTemplate(
                input_variables=["code", "problem_context", "ast_analysis", "pattern_analysis"],
                template="""You are a senior software engineer reviewing code during a technical interview.

CODE TO REVIEW:
```python
{code}
```

PROBLEM CONTEXT: {problem_context}
AST ANALYSIS: {ast_analysis}
DETECTED PATTERNS: {pattern_analysis}

Provide a constructive code review focusing on:
1. **Approach Assessment**: Is the overall approach sound?
2. **Code Quality**: Readability, structure, best practices
3. **Potential Issues**: Bugs, edge cases, inefficiencies
4. **Suggestions**: Specific improvements without giving away the solution

Keep feedback interview-appropriate: constructive but not overwhelming. Focus on the most important 2-3 points.

Code Review:"""
            )
            
            self.code_review_chain = LLMChain(
                llm=self.llm,
                prompt=code_review_prompt
            )
            
            # Approach analysis chain
            approach_analysis_prompt = PromptTemplate(
                input_variables=["code", "problem_type", "detected_patterns", "complexity_analysis"],
                template="""Analyze the coding approach being used for this problem.

CODE:
```python
{code}
```

PROBLEM TYPE: {problem_type}
DETECTED PATTERNS: {detected_patterns}
COMPLEXITY ANALYSIS: {complexity_analysis}

Analyze:
1. **Algorithm Choice**: What algorithm/approach is being used?
2. **Data Structures**: What data structures are employed?
3. **Time/Space Complexity**: What's the Big O analysis?
4. **Alternative Approaches**: Are there better approaches they could consider?
5. **Implementation Quality**: How well is the approach implemented?

Provide analysis that helps guide the candidate's thinking without spoiling the solution.

Approach Analysis:"""
            )
            
            self.approach_analysis_chain = LLMChain(
                llm=self.llm,
                prompt=approach_analysis_prompt
            )
            
            # Feedback generation chain
            feedback_prompt = PromptTemplate(
                input_variables=["analysis_results", "user_confidence_level", "interview_stage", "time_context"],
                template="""Generate appropriate feedback for a coding interview candidate.

ANALYSIS RESULTS: {analysis_results}
USER CONFIDENCE LEVEL: {user_confidence_level}
INTERVIEW STAGE: {interview_stage}
TIME CONTEXT: {time_context}

Generate feedback that:
1. Matches the user's confidence level (encouraging for low confidence, challenging for high confidence)
2. Is appropriate for the interview stage
3. Considers time constraints
4. Provides actionable guidance
5. Maintains positive interview atmosphere

Feedback should be conversational and supportive while being technically accurate.

Feedback:"""
            )
            
            self.feedback_generation_chain = LLMChain(
                llm=self.llm,
                prompt=feedback_prompt
            )
            
            logger.info("✅ LangChain chains initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up LangChain chains: {e}")
            raise
    
    async def _setup_crewai(self):
        """Setup CrewAI for complex code analysis orchestration"""
        try:
            # Define tools for code analysis
            tools = [
                Tool(
                    name="parse_ast",
                    description="Parse Python code into AST and extract structural information",
                    func=self._tool_parse_ast
                ),
                Tool(
                    name="detect_patterns",
                    description="Detect algorithmic patterns and approaches in code",
                    func=self._tool_detect_patterns
                ),
                Tool(
                    name="analyze_complexity",
                    description="Analyze time and space complexity of code",
                    func=self._tool_analyze_complexity
                ),
                Tool(
                    name="check_syntax",
                    description="Check code syntax and identify common errors",
                    func=self._tool_check_syntax
                ),
                Tool(
                    name="get_problem_context",
                    description="Get context about the current problem being solved",
                    func=self._tool_get_problem_context
                ),
                Tool(
                    name="assess_completeness",
                    description="Assess how complete the solution is",
                    func=self._tool_assess_completeness
                )
            ]
            
            # Create the code analyzer CrewAI agent
            self.crew_agent = Agent(
                role="Senior Code Review Specialist",
                goal="Provide insightful, constructive code analysis that helps candidates improve while maintaining interview integrity",
                backstory="""You are a senior software engineer with 10+ years of experience in code review 
                and technical mentoring. You have reviewed thousands of code solutions and are expert at:
                
                - Quickly identifying algorithmic approaches and their trade-offs
                - Spotting potential bugs and edge cases before they cause problems
                - Providing constructive feedback that guides learning without spoiling solutions
                - Recognizing when candidates are on the right track vs. need redirection
                - Balancing technical accuracy with interview psychology
                - Adapting feedback style to candidate confidence and experience level
                
                You understand that the goal is assessment AND education - helping candidates 
                demonstrate their best thinking while learning from the process.""",
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
            
            logger.info("✅ CrewAI code analyzer agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up CrewAI: {e}")
            raise
    
    def _load_algorithm_patterns(self) -> Dict[str, List[str]]:
        """Load algorithm pattern recognition data"""
        return {
            "two_pointers": [
                "left", "right", "while left < right", "start", "end",
                "i += 1", "j -= 1", "pointer"
            ],
            "sliding_window": [
                "window", "left", "right", "extend", "shrink",
                "window_start", "window_end", "max_length"
            ],
            "hash_map": [
                "dict()", "{}", "defaultdict", "Counter", "in dict",
                "get(", "setdefault", "hash", "lookup"
            ],
            "binary_search": [
                "left", "right", "mid", "// 2", "binary",
                "while left <= right", "search", "bisect"
            ],
            "dynamic_programming": [
                "dp", "memo", "cache", "subproblem", "recurrence",
                "bottom_up", "top_down", "memoization"
            ],
            "graph_traversal": [
                "visited", "queue", "stack", "dfs", "bfs",
                "neighbors", "adjacency", "graph"
            ],
            "sorting": [
                "sort", "sorted", "key=", "reverse=", "quicksort",
                "mergesort", "heapsort", "comparison"
            ],
            "greedy": [
                "greedy", "optimal", "local_optimum", "choice",
                "interval", "scheduling"
            ]
        }
    
    # ==========================================
    # CREWAI TOOLS FOR CODE ANALYSIS
    # ==========================================
    
    def _tool_parse_ast(self, code: str) -> str:
        """Parse Python code using AST"""
        try:
            if not code.strip():
                return json.dumps({"success": False, "error": "Empty code"})
            
            # Parse the AST
            tree = ast.parse(code)
            
            # Extract useful information
            functions = []
            variables = []
            imports = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    imports.append(f"from {node.module or ''}")
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            
            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            return json.dumps({
                "success": True,
                "functions": functions,
                "variables": list(set(variables)),
                "imports": imports,
                "classes": classes,
                "node_counts": dict(node_counts),
                "total_lines": code.count('\n') + 1,
                "complexity_score": len(list(ast.walk(tree)))
            })
            
        except SyntaxError as e:
            return json.dumps({
                "success": False,
                "error": "Syntax error",
                "details": str(e),
                "line": getattr(e, 'lineno', None),
                "offset": getattr(e, 'offset', None)
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"AST parsing error: {str(e)}"
            })
    
    def _tool_detect_patterns(self, code: str) -> str:
        """Detect algorithmic patterns in code"""
        try:
            if not code.strip():
                return json.dumps({"detected_patterns": [], "confidence": 0})
            
            code_lower = code.lower()
            detected_patterns = {}
            
            # Check for each pattern
            for pattern_name, keywords in self.algorithm_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in code_lower)
                if matches > 0:
                    confidence = min(matches / len(keywords), 1.0)
                    detected_patterns[pattern_name] = {
                        "confidence": confidence,
                        "matched_keywords": [kw for kw in keywords if kw in code_lower],
                        "match_count": matches
                    }
            
            # Sort by confidence
            sorted_patterns = sorted(
                detected_patterns.items(), 
                key=lambda x: x[1]["confidence"], 
                reverse=True
            )
            
            return json.dumps({
                "detected_patterns": dict(sorted_patterns),
                "primary_pattern": sorted_patterns[0][0] if sorted_patterns else None,
                "pattern_count": len(detected_patterns),
                "analysis": f"Detected {len(detected_patterns)} potential algorithmic patterns"
            })
            
        except Exception as e:
            return json.dumps({
                "detected_patterns": {},
                "error": f"Pattern detection error: {str(e)}"
            })
    
    def _tool_analyze_complexity(self, code: str) -> str:
        """Analyze time and space complexity"""
        try:
            if not code.strip():
                return json.dumps({"time_complexity": "O(?)", "space_complexity": "O(?)"})
            
            # Parse AST for complexity analysis
            tree = ast.parse(code)
            
            # Count nested loops and operations
            loop_depth = 0
            max_loop_depth = 0
            recursive_calls = 0
            data_structures = []
            
            class ComplexityAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.current_depth = 0
                    self.max_depth = 0
                    self.has_recursion = False
                    self.data_structures = set()
                
                def visit_For(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_While(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        # Check for recursive calls (simplified)
                        if hasattr(node.func, 'id'):
                            self.data_structures.add(node.func.id)
                    self.generic_visit(node)
                
                def visit_List(self, node):
                    self.data_structures.add("list")
                    self.generic_visit(node)
                
                def visit_Dict(self, node):
                    self.data_structures.add("dict")
                    self.generic_visit(node)
                
                def visit_Set(self, node):
                    self.data_structures.add("set")
                    self.generic_visit(node)
            
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            
            # Estimate complexity based on structure
            if analyzer.max_depth == 0:
                time_complexity = "O(1)"
            elif analyzer.max_depth == 1:
                time_complexity = "O(n)"
            elif analyzer.max_depth == 2:
                time_complexity = "O(n²)"
            else:
                time_complexity = f"O(n^{analyzer.max_depth})"
            
            # Check for specific patterns that affect complexity
            if "sort" in code.lower():
                time_complexity = "O(n log n)"
            elif "binary" in code.lower() or "// 2" in code:
                time_complexity = "O(log n)"
            
            # Space complexity estimation
            if "dict" in analyzer.data_structures or "set" in analyzer.data_structures:
                space_complexity = "O(n)"
            elif "list" in analyzer.data_structures:
                space_complexity = "O(n)"
            else:
                space_complexity = "O(1)"
            
            return json.dumps({
                "time_complexity": time_complexity,
                "space_complexity": space_complexity,
                "loop_depth": analyzer.max_depth,
                "data_structures": list(analyzer.data_structures),
                "analysis": f"Max loop depth: {analyzer.max_depth}, Data structures: {list(analyzer.data_structures)}"
            })
            
        except Exception as e:
            return json.dumps({
                "time_complexity": "O(?)",
                "space_complexity": "O(?)",
                "error": f"Complexity analysis error: {str(e)}"
            })
    
    def _tool_check_syntax(self, code: str) -> str:
        """Check code syntax and common errors"""
        try:
            if not code.strip():
                return json.dumps({"syntax_valid": False, "error": "Empty code"})
            
            # Try to parse the code
            try:
                ast.parse(code)
                syntax_valid = True
                syntax_error = None
            except SyntaxError as e:
                syntax_valid = False
                syntax_error = {
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset,
                    "text": e.text
                }
            
            # Check for common issues
            issues = []
            
            # Check for indentation issues
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                if line.strip() and line.startswith(' ') and '\t' in line:
                    issues.append(f"Line {i}: Mixed tabs and spaces")
            
            # Check for common Python issues
            if 'print ' in code:  # Python 2 style print
                issues.append("Using Python 2 style print statements")
            
            if '= =' in code or '= <' in code or '= >' in code:
                issues.append("Possible assignment instead of comparison")
            
            # Check for missing colons
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if (stripped.startswith(('if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ')) and 
                    not stripped.endswith(':')):
                    issues.append(f"Line {i}: Missing colon")
            
            return json.dumps({
                "syntax_valid": syntax_valid,
                "syntax_error": syntax_error,
                "issues": issues,
                "line_count": len(lines),
                "analysis": "Syntax check completed"
            })
            
        except Exception as e:
            return json.dumps({
                "syntax_valid": False,
                "error": f"Syntax check error: {str(e)}"
            })
    
    def _tool_get_problem_context(self, problem_id: str) -> str:
        """Get context about the current problem"""
        try:
            if not problem_id:
                return json.dumps({"success": False, "error": "No problem ID provided"})
            
            with self._db_lock:
                conn = self.sync_db_pool.getconn()
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute(
                        "SELECT title, description, difficulty, topics, test_cases FROM problems WHERE id = %s",
                        (problem_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        return json.dumps({
                            "success": True,
                            "title": result['title'],
                            "description": result['description'],
                            "difficulty": result['difficulty'],
                            "topics": result['topics'] or [],
                            "test_cases": result['test_cases'] or [],
                            "analysis": f"Problem context for {result['title']}"
                        })
                    else:
                        return json.dumps({
                            "success": False,
                            "error": "Problem not found"
                        })
                        
                finally:
                    self.sync_db_pool.putconn(conn)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Database error: {str(e)}"
            })
    
    def _tool_assess_completeness(self, code: str, problem_context: str = "") -> str:
        """Assess how complete the solution is"""
        try:
            if not code.strip():
                return json.dumps({"completeness": 0, "status": "empty"})
            
            # Parse problem context
            problem_data = {}
            if problem_context:
                try:
                    problem_data = json.loads(problem_context)
                except:
                    pass
            
            # Analyze code structure
            try:
                tree = ast.parse(code)
                has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
                has_logic = any(isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(tree))
                
                # Calculate completeness score
                completeness_score = 0
                status_indicators = []
                
                if has_function:
                    completeness_score += 30
                    status_indicators.append("has_function")
                
                if has_return:
                    completeness_score += 25
                    status_indicators.append("has_return")
                
                if has_logic:
                    completeness_score += 25
                    status_indicators.append("has_logic")
                
                # Check for pass statements (incomplete code)
                if 'pass' in code:
                    completeness_score = max(0, completeness_score - 20)
                    status_indicators.append("has_pass")
                
                # Check for TODO comments
                if 'TODO' in code.upper() or '#' in code:
                    completeness_score += 5  # Shows planning
                    status_indicators.append("has_comments")
                
                # Bonus for apparent completeness
                if completeness_score >= 80 and len(code.strip()) > 50:
                    completeness_score = min(100, completeness_score + 20)
                
                # Determine status
                if completeness_score < 25:
                    status = "skeleton"
                elif completeness_score < 50:
                    status = "partial"
                elif completeness_score < 80:
                    status = "substantial"
                else:
                    status = "complete"
                
                return json.dumps({
                    "completeness": completeness_score,
                    "status": status,
                    "indicators": status_indicators,
                    "has_function": has_function,
                    "has_return": has_return,
                    "has_logic": has_logic,
                    "analysis": f"Code appears {status} ({completeness_score}% complete)"
                })
                
            except SyntaxError:
                return json.dumps({
                    "completeness": 10,
                    "status": "syntax_error",
                    "analysis": "Code has syntax errors"
                })
                
        except Exception as e:
            return json.dumps({
                "completeness": 0,
                "status": "error",
                "error": f"Assessment error: {str(e)}"
            })
    
    # ==========================================
    # MESSAGE HANDLERS
    # ==========================================
    
    async def handle_code_analysis(self, message: AgentMessage):
        """Handle code analysis requests"""
        try:
            # Only handle messages targeting code analyzer
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            action = payload.get("action", "analyze_code")
            session_id = payload.get("session_id", "")
            
            if not session_id:
                logger.warning("No session_id in code analysis message")
                return
            
            # Route to appropriate handler based on action
            if action == "analyze_code" or action == "code_change":
                await self._handle_code_analysis(session_id, payload)
            elif action == "analyze_input":
                await self._handle_input_analysis(session_id, payload)
            elif action == "review_solution":
                await self._handle_solution_review(session_id, payload)
            else:
                logger.warning(f"Unknown action for code analyzer: {action}")
                await self._handle_code_analysis(session_id, payload)
                
        except Exception as e:
            logger.error(f"❌ Error handling code analysis: {e}")
    
    async def handle_user_interaction(self, message: AgentMessage):
        """Handle user interaction messages routed to code analyzer"""
        try:
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            action = payload.get("action", "analyze_input")
            
            if action == "analyze_input":
                await self._handle_input_analysis(payload.get("session_id", ""), payload)
                
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
    # CODE ANALYSIS HANDLERS
    # ==========================================
    
    async def _handle_code_analysis(self, session_id: str, payload: Dict[str, Any]):
        """Handle comprehensive code analysis"""
        try:
            code = payload.get("code", "")
            problem_id = payload.get("problem_id", "")
            
            if not code.strip():
                logger.info("Empty code received for analysis")
                return
            
            logger.info(f"🔍 Analyzing code for session {session_id}: {len(code)} characters")
            
            # Use CrewAI for comprehensive analysis
            task = Task(
                description=f"""
                Perform comprehensive analysis of this Python code:
                
                ```python
                {code}
                ```
                
                Problem ID: {problem_id}
                Session: {session_id}
                
                Use your tools to:
                1. Parse the AST and extract structural information
                2. Detect algorithmic patterns and approaches
                3. Analyze time/space complexity
                4. Check syntax and identify issues
                5. Get problem context for relevance assessment
                6. Assess solution completeness
                
                Provide a thorough but constructive analysis that helps the candidate improve their code
                while maintaining interview integrity. Focus on the most important insights.
                """,
                agent=self.crew_agent,
                expected_output="Comprehensive code analysis with actionable feedback"
            )
            
            # Execute the analysis
            analysis_result = str(self.crew.kickoff(tasks=[task]))
            
            # Cache the analysis
            self.analysis_cache[session_id] = {
                "code": code,
                "analysis": analysis_result,
                "timestamp": datetime.now(),
                "problem_id": problem_id
            }
            
            # Send analysis to coordinator/interviewer
            await self._send_analysis_response(session_id, analysis_result, "code_analysis")
            
            logger.info(f"✅ Code analysis completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error in code analysis: {e}")
            await self._send_analysis_response(
                session_id,
                "I encountered an issue analyzing your code. Could you check for any syntax errors and try again?",
                "analysis_error"
            )
    
    async def _handle_input_analysis(self, session_id: str, payload: Dict[str, Any]):
        """Handle analysis of user input about code"""
        try:
            user_content = payload.get("content", "")
            
            # Check if this is code-related input
            if not self._is_code_related_input(user_content):
                # Route back to interviewer
                await self._route_to_interviewer(session_id, user_content)
                return
            
            logger.info(f"🤔 Analyzing code-related input for session {session_id}")
            
            # Get cached analysis if available
            cached_analysis = self.analysis_cache.get(session_id, {})
            current_code = cached_analysis.get("code", "")
            
            # Use CrewAI to analyze the input in context
            task = Task(
                description=f"""
                The candidate made this code-related comment: "{user_content}"
                
                Current code context: {current_code[:500] if current_code else "No code yet"}
                Cached analysis: {cached_analysis.get('analysis', 'None')[:300] if cached_analysis else "None"}
                
                Analyze their input and provide appropriate technical guidance:
                1. What are they trying to communicate about their code?
                2. Do they need clarification, validation, or technical help?
                3. What's the best way to respond to guide their coding process?
                
                Provide a helpful response that addresses their technical concern while maintaining
                interview standards (guide, don't solve).
                """,
                agent=self.crew_agent,
                expected_output="Technical guidance response to user's code-related input"
            )
            
            response = str(self.crew.kickoff(tasks=[task]))
            
            # Send response
            await self._send_analysis_response(session_id, response, "input_analysis")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing user input: {e}")
            await self._route_to_interviewer(session_id, user_content)
    
    async def _handle_solution_review(self, session_id: str, payload: Dict[str, Any]):
        """Handle final solution review"""
        try:
            code = payload.get("code", "")
            problem_context = payload.get("problem_context", {})
            
            logger.info(f"📋 Reviewing solution for session {session_id}")
            
            # Comprehensive solution review using LangChain chains
            
            # 1. Get AST analysis
            ast_result = self._tool_parse_ast(code)
            
            # 2. Detect patterns
            pattern_result = self._tool_detect_patterns(code)
            
            # 3. Analyze complexity
            complexity_result = self._tool_analyze_complexity(code)
            
            # 4. Assess completeness
            completeness_result = self._tool_assess_completeness(code, json.dumps(problem_context))
            
            # 5. Generate comprehensive review
            review_result = await self.code_review_chain.arun(
                code=code,
                problem_context=json.dumps(problem_context),
                ast_analysis=ast_result,
                pattern_analysis=pattern_result
            )
            
            # 6. Generate approach analysis
            approach_result = await self.approach_analysis_chain.arun(
                code=code,
                problem_type=problem_context.get("title", "Coding Problem"),
                detected_patterns=pattern_result,
                complexity_analysis=complexity_result
            )
            
            # Combine results
            final_review = f"""**Solution Review:**

{review_result}

**Approach Analysis:**

{approach_result}

**Technical Details:**
- Complexity: {json.loads(complexity_result).get('time_complexity', 'Unknown')} time, {json.loads(complexity_result).get('space_complexity', 'Unknown')} space
- Completeness: {json.loads(completeness_result).get('completeness', 0)}%
- Primary Pattern: {json.loads(pattern_result).get('primary_pattern', 'None detected')}"""
            
            await self._send_analysis_response(session_id, final_review, "solution_review")
            
        except Exception as e:
            logger.error(f"❌ Error in solution review: {e}")
            await self._send_analysis_response(
                session_id,
                "I'll review your solution. The approach looks reasonable - let me analyze the implementation details.",
                "review_fallback"
            )
    
    def _is_code_related_input(self, user_input: str) -> bool:
        """Determine if user input is code-related"""
        code_keywords = [
            "algorithm", "approach", "implementation", "solution", "code", "function",
            "variable", "loop", "recursion", "complexity", "optimize", "efficient",
            "bug", "error", "syntax", "logic", "debug", "test", "output"
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in code_keywords)
    
    async def _route_to_interviewer(self, session_id: str, user_content: str):
        """Route non-code input back to interviewer"""
        try:
            route_message = AgentMessage(
                event_type=EventType.AGENT_RESPONSE,
                source_agent=self.agent_name,
                target_agent="interviewer",
                payload={
                    "action": "handle_user_input",
                    "content": user_content,
                    "session_id": session_id,
                    "routed_from": "code_analyzer"
                },
                context_snapshot={}
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, route_message)
            logger.info(f"📤 Routed non-code input to interviewer: {user_content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error routing to interviewer: {e}")
    
    async def _send_analysis_response(self, session_id: str, content: str, response_type: str = "analysis"):
        """Send analysis response back to user"""
        try:
            response_message = AgentMessage(
                event_type=EventType.AGENT_RESPONSE,
                source_agent=self.agent_name,
                target_agent="user",
                payload={
                    "speaker": "code_analyzer",
                    "content": content,
                    "session_id": session_id,
                    "response_type": response_type,
                    "timestamp": datetime.now().isoformat()
                },
                context_snapshot={}
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, response_message)
            logger.info(f"📤 Sent analysis response ({response_type}): {content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error sending analysis response: {e}")
    
    # ==========================================
    # MAIN RUN LOOP
    # ==========================================
    
    async def run(self):
        """Main code analyzer run loop"""
        self.running = True
        logger.info("🚀 Code Analyzer Agent starting...")
        
        try:
            await self.initialize()
            
            # Update status to active
            await self.message_bus.register_agent(self.agent_name, {
                "type": "code_analyzer",
                "capabilities": ["ast_analysis", "pattern_recognition", "code_review", "approach_detection"],
                "status": "active"
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start listening for messages
            await self.message_bus.start_listening()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in code analyzer: {e}")
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
        logger.info("🛑 Code Analyzer shutting down...")
        self.running = False
        
        if self.message_bus:
            await self.message_bus.shutdown()
        if self.sync_db_pool:
            self.sync_db_pool.closeall()

# Entry point
async def main():
    analyzer = CodeAnalyzerAgent()
    try:
        await analyzer.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await analyzer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())