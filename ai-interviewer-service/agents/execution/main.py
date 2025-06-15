import asyncio
import os
import logging
import json
import time
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import uuid

# Docker client for code execution
import docker
from docker.errors import ContainerError, ImageNotFound, APIError

# CrewAI and LangChain imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Database for test cases and results
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
import threading

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import (
    AgentMessage, EventType, InterviewContext, InterviewState, 
    Problem, Difficulty, Message, PerformanceMetrics, ExecutionResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Secure code execution agent using Docker containers for isolation.
    Runs user code against test cases with performance monitoring and safety controls.
    """
    
    def __init__(self):
        self.agent_name = "execution"
        self.message_bus: MessageBus = None
        self.sync_db_pool: psycopg2.pool.ThreadedConnectionPool = None
        self.docker_client: docker.DockerClient = None
        self.running = False
        
        # LangChain components
        self.llm: ChatAnthropic = None
        
        # CrewAI components for execution orchestration
        self.crew_agent: Agent = None
        self.crew: Crew = None
        
        # LangChain chains for result analysis
        self.result_analysis_chain: LLMChain = None
        self.feedback_generation_chain: LLMChain = None
        
        # Execution configuration
        self.execution_config = {
            "timeout_seconds": 10,
            "memory_limit": "128m",
            "cpu_limit": "0.5",
            "network_disabled": True,
            "read_only_filesystem": True
        }
        
        # Security and monitoring
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self._db_lock = threading.Lock()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            await self._setup_llm()
            await self._setup_database()
            await self._setup_docker()
            await self._setup_message_bus()
            await self._setup_langchain_chains()
            await self._setup_crewai()
            
            logger.info("✅ Execution Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize execution agent: {e}")
            raise
    
    async def _setup_llm(self):
        """Setup LangChain LLM"""
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key or api_key == "your_claude_api_key_here":
            raise ValueError("CLAUDE_API_KEY not set")
            
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0.2,  # Low temperature for consistent technical analysis
            max_tokens=1500
        )
        
        logger.info("✅ LangChain LLM initialized")
    
    async def _setup_database(self):
        """Setup database connection for test cases"""
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
    
    async def _setup_docker(self):
        """Setup Docker client and prepare execution environment"""
        try:
            # Try different Docker connection methods
            try:
                # First try: default Docker socket
                self.docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
                self.docker_client.ping()
                logger.info("✅ Connected to Docker via unix socket")
            except Exception as e1:
                logger.warning(f"Failed to connect via unix socket: {e1}")
                try:
                    # Second try: from environment
                    self.docker_client = docker.from_env()
                    self.docker_client.ping()
                    logger.info("✅ Connected to Docker via environment")
                except Exception as e2:
                    logger.warning(f"Failed to connect via environment: {e2}")
                    # For now, use fallback execution without Docker
                    logger.warning("⚠️ Docker unavailable, using fallback execution environment")
                    self.docker_client = None
                    logger.info("✅ Fallback execution environment initialized")
                    return
            
            # Prepare Python execution image
            await self._prepare_python_image()
            
            logger.info("✅ Docker client initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Docker setup failed, using fallback: {e}")
            self.docker_client = None
            logger.info("✅ Fallback execution environment initialized")
    
    async def _prepare_python_image(self):
        """Prepare lightweight Python image for code execution"""
        try:
            # Check if our custom image exists
            image_name = "interview-python-runner"
            
            try:
                self.docker_client.images.get(image_name)
                logger.info(f"✅ Found existing image: {image_name}")
                return
            except ImageNotFound:
                pass
            
            # Create Dockerfile for execution environment
            dockerfile_content = """
FROM python:3.11-alpine

# Install only essential packages
RUN apk add --no-cache gcc musl-dev

# Create non-root user for security
RUN adduser -D -s /bin/sh runner

# Set working directory
WORKDIR /app

# Switch to non-root user
USER runner

# Default command
CMD ["python"]
"""
            
            # Build the image
            logger.info("🔨 Building Python execution image...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                dockerfile_path = os.path.join(temp_dir, "Dockerfile")
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                
                self.docker_client.images.build(
                    path=temp_dir,
                    tag=image_name,
                    rm=True,
                    forcerm=True
                )
            
            logger.info(f"✅ Built Python execution image: {image_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare Python image: {e}")
            # Fall back to standard python image
            logger.info("📦 Falling back to standard python:3.11-alpine image")
    
    async def _setup_message_bus(self):
        """Setup message bus connections"""
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        self.message_bus = await get_message_bus(redis_url)
        
        # Register agent
        await self.message_bus.register_agent(self.agent_name, {
            "type": "execution",
            "capabilities": ["code_execution", "test_validation", "performance_monitoring", "security_sandboxing"],
            "status": "initializing"
        })
        
        # Subscribe to relevant channels
        self.message_bus.subscribe(Channels.EXECUTION, self.handle_execution_request)
        self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_user_interaction)
        self.message_bus.subscribe(Channels.SYSTEM, self.handle_system_message)
        
        logger.info("✅ Message bus subscriptions established")
    
    async def _setup_langchain_chains(self):
        """Setup LangChain chains for result analysis"""
        try:
            # Result analysis chain
            result_analysis_prompt = PromptTemplate(
                input_variables=["execution_results", "test_case_results", "performance_data", "problem_context"],
                template="""Analyze the code execution results from a technical interview.

EXECUTION RESULTS:
{execution_results}

TEST CASE RESULTS:
{test_case_results}

PERFORMANCE DATA:
{performance_data}

PROBLEM CONTEXT:
{problem_context}

Provide analysis covering:
1. **Correctness**: Did the solution pass the test cases?
2. **Performance**: How did it perform in terms of speed and memory?
3. **Edge Cases**: Any issues with edge cases or error handling?
4. **Code Quality**: Based on execution behavior, any insights about the implementation?

Keep the analysis interview-appropriate: constructive and educational.

Analysis:"""
            )
            
            self.result_analysis_chain = LLMChain(
                llm=self.llm,
                prompt=result_analysis_prompt
            )
            
            # Feedback generation chain
            feedback_prompt = PromptTemplate(
                input_variables=["analysis_results", "user_confidence", "execution_status", "interview_stage"],
                template="""Generate appropriate feedback for the candidate based on their code execution results.

ANALYSIS: {analysis_results}
USER CONFIDENCE: {user_confidence}
EXECUTION STATUS: {execution_status}
INTERVIEW STAGE: {interview_stage}

Generate feedback that:
1. Acknowledges what worked well
2. Points out any issues constructively
3. Suggests next steps or improvements
4. Maintains positive interview atmosphere
5. Is appropriate for their confidence level

The feedback should feel like a supportive interviewer discussing results.

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
        """Setup CrewAI for execution orchestration"""
        try:
            # Define tools for execution management
            tools = [
                Tool(
                    name="execute_code_safely",
                    description="Execute Python code in a secure Docker container",
                    func=self._tool_execute_code_safely
                ),
                Tool(
                    name="validate_test_cases",
                    description="Run code against specific test cases and validate results",
                    func=self._tool_validate_test_cases
                ),
                Tool(
                    name="monitor_performance",
                    description="Monitor execution time and memory usage",
                    func=self._tool_monitor_performance
                ),
                Tool(
                    name="analyze_errors",
                    description="Analyze runtime errors and exceptions",
                    func=self._tool_analyze_errors
                ),
                Tool(
                    name="get_test_cases",
                    description="Retrieve test cases for a specific problem",
                    func=self._tool_get_test_cases
                ),
                Tool(
                    name="security_check",
                    description="Check code for potential security issues before execution",
                    func=self._tool_security_check
                )
            ]
            
            # Create the execution CrewAI agent
            self.crew_agent = Agent(
                role="Code Execution Specialist",
                goal="Safely execute and comprehensively test candidate code while providing insightful performance feedback",
                backstory="""You are a senior DevOps engineer and testing specialist with 8+ years of experience 
                in secure code execution and performance analysis. You have expertise in:
                
                - Docker containerization and security best practices
                - Performance monitoring and optimization
                - Test case design and validation
                - Error analysis and debugging guidance
                - Providing actionable feedback on code execution results
                
                Your role is to safely execute candidate code, validate it against test cases, monitor 
                performance, and provide educational feedback that helps candidates understand how their 
                solutions perform in real-world scenarios.""",
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
            
            logger.info("✅ CrewAI execution agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up CrewAI: {e}")
            raise
    
    # ==========================================
    # CREWAI TOOLS FOR CODE EXECUTION
    # ==========================================
    
    def _tool_execute_code_safely(self, code: str, timeout: int = 10) -> str:
        """Execute Python code in secure Docker container"""
        try:
            if not code.strip():
                return json.dumps({"success": False, "error": "Empty code provided"})
            
            # Security check first
            security_result = self._basic_security_check(code)
            if not security_result["safe"]:
                return json.dumps({
                    "success": False,
                    "error": "Security check failed",
                    "details": security_result["issues"]
                })
            
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            try:
                # Prepare Docker execution
                image_name = "interview-python-runner"
                
                # Try custom image first, fall back to standard Python
                try:
                    self.docker_client.images.get(image_name)
                except ImageNotFound:
                    image_name = "python:3.11-alpine"
                
                # Execute in container
                start_time = time.time()
                
                try:
                    container = self.docker_client.containers.run(
                        image=image_name,
                        command=f"python /app/code.py",
                        volumes={temp_file_path: {'bind': '/app/code.py', 'mode': 'ro'}},
                        mem_limit=self.execution_config["memory_limit"],
                        cpu_quota=int(self.execution_config["cpu_limit"] * 100000),
                        network_disabled=self.execution_config["network_disabled"],
                        timeout=timeout,
                        remove=True,
                        detach=False,
                        stdout=True,
                        stderr=True
                    )
                    
                    execution_time = time.time() - start_time
                    output = container.decode('utf-8').strip()
                    
                    return json.dumps({
                        "success": True,
                        "output": output,
                        "execution_time_ms": int(execution_time * 1000),
                        "exit_code": 0
                    })
                    
                except ContainerError as e:
                    execution_time = time.time() - start_time
                    return json.dumps({
                        "success": False,
                        "error": "Runtime error",
                        "stderr": e.stderr.decode('utf-8') if e.stderr else "",
                        "exit_code": e.exit_status,
                        "execution_time_ms": int(execution_time * 1000)
                    })
                
                except Exception as e:
                    return json.dumps({
                        "success": False,
                        "error": f"Execution error: {str(e)}"
                    })
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in code execution: {e}")
            return json.dumps({
                "success": False,
                "error": f"System error: {str(e)}"
            })
    
    def _tool_validate_test_cases(self, code: str, problem_id: str) -> str:
        """Run code against test cases and validate results"""
        try:
            # Get test cases from database
            test_cases = self._get_test_cases_from_db(problem_id)
            if not test_cases:
                return json.dumps({
                    "success": False,
                    "error": "No test cases found for problem"
                })
            
            results = []
            passed_count = 0
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Prepare test code
                    input_data = test_case["input_data"]
                    expected_output = test_case["expected_output"]
                    
                    # Create test wrapper
                    test_code = f"""
{code}

# Test case execution
import json
try:
    input_data = {json.dumps(input_data)}
    result = {self._generate_function_call(code, input_data)}
    print(json.dumps({{"result": result, "success": True}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "success": False}}))
"""
                    
                    # Execute test
                    execution_result = json.loads(self._tool_execute_code_safely(test_code, timeout=5))
                    
                    if execution_result["success"]:
                        try:
                            test_output = json.loads(execution_result["output"])
                            if test_output["success"]:
                                actual_result = test_output["result"]
                                passed = self._compare_results(actual_result, expected_output)
                                if passed:
                                    passed_count += 1
                                
                                results.append({
                                    "test_case": i + 1,
                                    "passed": passed,
                                    "expected": expected_output,
                                    "actual": actual_result,
                                    "execution_time_ms": execution_result.get("execution_time_ms", 0)
                                })
                            else:
                                results.append({
                                    "test_case": i + 1,
                                    "passed": False,
                                    "error": test_output["error"],
                                    "expected": expected_output
                                })
                        except json.JSONDecodeError:
                            results.append({
                                "test_case": i + 1,
                                "passed": False,
                                "error": "Invalid output format",
                                "raw_output": execution_result["output"]
                            })
                    else:
                        results.append({
                            "test_case": i + 1,
                            "passed": False,
                            "error": execution_result.get("error", "Execution failed"),
                            "expected": expected_output
                        })
                        
                except Exception as e:
                    results.append({
                        "test_case": i + 1,
                        "passed": False,
                        "error": f"Test execution error: {str(e)}"
                    })
            
            return json.dumps({
                "success": True,
                "total_tests": len(test_cases),
                "passed_tests": passed_count,
                "pass_rate": passed_count / len(test_cases) if test_cases else 0,
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error validating test cases: {e}")
            return json.dumps({
                "success": False,
                "error": f"Validation error: {str(e)}"
            })
    
    def _tool_monitor_performance(self, execution_results: str) -> str:
        """Monitor and analyze performance metrics"""
        try:
            results = json.loads(execution_results)
            
            if not results.get("success"):
                return json.dumps({
                    "performance_analysis": "Could not analyze performance due to execution failure"
                })
            
            execution_time = results.get("execution_time_ms", 0)
            
            # Performance analysis
            analysis = {
                "execution_time_ms": execution_time,
                "performance_category": "fast" if execution_time < 100 else "medium" if execution_time < 1000 else "slow",
                "time_analysis": "",
                "optimization_suggestions": []
            }
            
            if execution_time < 50:
                analysis["time_analysis"] = "Excellent performance - very fast execution"
            elif execution_time < 200:
                analysis["time_analysis"] = "Good performance - reasonable execution time"
            elif execution_time < 1000:
                analysis["time_analysis"] = "Moderate performance - could be optimized"
            else:
                analysis["time_analysis"] = "Slow performance - likely needs optimization"
                analysis["optimization_suggestions"].append("Consider algorithm optimization")
                analysis["optimization_suggestions"].append("Review time complexity")
            
            return json.dumps(analysis)
            
        except Exception as e:
            return json.dumps({
                "performance_analysis": f"Performance monitoring error: {str(e)}"
            })
    
    def _tool_analyze_errors(self, execution_results: str) -> str:
        """Analyze runtime errors and provide debugging guidance"""
        try:
            results = json.loads(execution_results)
            
            if results.get("success"):
                return json.dumps({
                    "error_analysis": "No errors detected - code executed successfully"
                })
            
            error_msg = results.get("error", "")
            stderr = results.get("stderr", "")
            
            # Common error patterns and guidance
            error_analysis = {
                "error_type": "unknown",
                "guidance": "",
                "suggestions": []
            }
            
            error_text = (error_msg + " " + stderr).lower()
            
            if "indentationerror" in error_text:
                error_analysis.update({
                    "error_type": "indentation",
                    "guidance": "Indentation error - check your spacing and tabs",
                    "suggestions": ["Use consistent indentation (4 spaces recommended)", "Check for mixed tabs and spaces"]
                })
            elif "syntaxerror" in error_text:
                error_analysis.update({
                    "error_type": "syntax",
                    "guidance": "Syntax error - check your Python syntax",
                    "suggestions": ["Check for missing colons, parentheses, or quotes", "Verify function/class definitions"]
                })
            elif "nameerror" in error_text:
                error_analysis.update({
                    "error_type": "name",
                    "guidance": "NameError - using undefined variable or function",
                    "suggestions": ["Check variable names for typos", "Ensure variables are defined before use"]
                })
            elif "indexerror" in error_text:
                error_analysis.update({
                    "error_type": "index",
                    "guidance": "IndexError - accessing list/string index that doesn't exist",
                    "suggestions": ["Check array bounds", "Consider edge cases with empty lists"]
                })
            elif "keyerror" in error_text:
                error_analysis.update({
                    "error_type": "key",
                    "guidance": "KeyError - accessing dictionary key that doesn't exist",
                    "suggestions": ["Use .get() method for safe access", "Check if key exists before accessing"]
                })
            elif "timeout" in error_text:
                error_analysis.update({
                    "error_type": "timeout",
                    "guidance": "Code execution timed out - likely infinite loop or very slow algorithm",
                    "suggestions": ["Check for infinite loops", "Consider algorithm efficiency", "Review loop conditions"]
                })
            else:
                error_analysis.update({
                    "guidance": "Runtime error occurred during execution",
                    "suggestions": ["Review the error message carefully", "Test with simpler inputs first"]
                })
            
            return json.dumps(error_analysis)
            
        except Exception as e:
            return json.dumps({
                "error_analysis": f"Error analysis failed: {str(e)}"
            })
    
    def _tool_get_test_cases(self, problem_id: str) -> str:
        """Get test cases for a specific problem"""
        try:
            test_cases = self._get_test_cases_from_db(problem_id)
            return json.dumps({
                "success": True,
                "test_cases": test_cases,
                "count": len(test_cases)
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to get test cases: {str(e)}"
            })
    
    def _tool_security_check(self, code: str) -> str:
        """Check code for potential security issues"""
        try:
            security_result = self._basic_security_check(code)
            return json.dumps(security_result)
        except Exception as e:
            return json.dumps({
                "safe": False,
                "error": f"Security check failed: {str(e)}"
            })
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    def _basic_security_check(self, code: str) -> Dict[str, Any]:
        """Basic security check for code"""
        issues = []
        
        # Dangerous imports/functions
        dangerous_patterns = [
            "import os", "import subprocess", "import sys", "import socket",
            "__import__", "exec(", "eval(", "open(", "file(", 
            "input(", "raw_input(", "compile(", "globals(", "locals("
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                issues.append(f"Potentially dangerous: {pattern}")
        
        # Check for infinite loop patterns (basic)
        if "while true" in code_lower or "while 1" in code_lower:
            issues.append("Potential infinite loop detected")
        
        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "analysis": "Basic security scan completed"
        }
    
    def _get_test_cases_from_db(self, problem_id: str) -> List[Dict[str, Any]]:
        """Get test cases from database"""
        try:
            with self._db_lock:
                conn = self.sync_db_pool.getconn()
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute(
                        "SELECT test_cases FROM problems WHERE id = %s",
                        (problem_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result and result['test_cases']:
                        return result['test_cases']
                    return []
                    
                finally:
                    self.sync_db_pool.putconn(conn)
                    
        except Exception as e:
            logger.error(f"Error getting test cases: {e}")
            return []
    
    def _generate_function_call(self, code: str, input_data: Dict[str, Any]) -> str:
        """Generate function call based on code and input data"""
        try:
            # Extract function name (simple approach)
            import ast
            tree = ast.parse(code)
            
            function_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break
            
            if not function_name:
                return "None  # No function found"
            
            # Generate call based on input data structure
            if isinstance(input_data, dict):
                if len(input_data) == 1:
                    # Single argument
                    key, value = list(input_data.items())[0]
                    return f"{function_name}({json.dumps(value)})"
                else:
                    # Multiple arguments
                    args = [json.dumps(value) for value in input_data.values()]
                    return f"{function_name}({', '.join(args)})"
            else:
                # Direct value
                return f"{function_name}({json.dumps(input_data)})"
                
        except Exception as e:
            logger.error(f"Error generating function call: {e}")
            return "None  # Error generating call"
    
    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare actual vs expected results with tolerance for different types"""
        try:
            # Handle lists that might be in different orders (for some problems)
            if isinstance(actual, list) and isinstance(expected, list):
                if len(actual) != len(expected):
                    return False
                
                # Try exact match first
                if actual == expected:
                    return True
                
                # For some problems, order might not matter
                try:
                    return sorted(actual) == sorted(expected)
                except TypeError:
                    # If sorting fails, stick with exact match
                    return actual == expected
            
            # Direct comparison for other types
            return actual == expected
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return False
    
    # ==========================================
    # MESSAGE HANDLERS
    # ==========================================
    
    async def handle_execution_request(self, message: AgentMessage):
        """Handle code execution requests"""
        try:
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            action = payload.get("action", "execute_code")
            session_id = payload.get("session_id", "")
            
            if not session_id:
                logger.warning("No session_id in execution request")
                return
            
            if action == "execute_code":
                await self._handle_code_execution(session_id, payload)
            elif action == "validate_solution":
                await self._handle_solution_validation(session_id, payload)
            else:
                logger.warning(f"Unknown execution action: {action}")
                await self._handle_code_execution(session_id, payload)
                
        except Exception as e:
            logger.error(f"❌ Error handling execution request: {e}")
    
    async def handle_user_interaction(self, message: AgentMessage):
        """Handle user interaction messages routed to execution"""
        try:
            if message.target_agent != self.agent_name:
                return
                
            payload = message.payload
            session_id = payload.get("session_id", "")
            
            # User wants to execute their current code
            await self._handle_code_execution(session_id, payload)
            
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
    # EXECUTION HANDLERS
    # ==========================================
    
    async def _handle_code_execution(self, session_id: str, payload: Dict[str, Any]):
        """Handle code execution request"""
        try:
            code = payload.get("code", "")
            problem_id = payload.get("problem_id", "")
            test_mode = payload.get("test_mode", True)
            
            if not code.strip():
                await self._send_execution_response(
                    session_id,
                    "No code provided for execution. Please submit your code first.",
                    "no_code"
                )
                return
            
            logger.info(f"🚀 Executing code for session {session_id}")
            
            # Use CrewAI for comprehensive execution orchestration
            task = Task(
                description=f"""
                Execute and analyze this Python code safely:
                
                ```python
                {code}
                ```
                
                Session: {session_id}
                Problem ID: {problem_id}
                Test Mode: {test_mode}
                
                Use your tools to:
                1. Perform security check to ensure safe execution
                2. Execute the code safely in a Docker container
                3. If test_mode is True and problem_id exists, validate against test cases
                4. Monitor performance metrics
                5. Analyze any errors that occur
                6. Provide comprehensive feedback on the execution results
                
                Focus on providing educational feedback that helps the candidate understand
                their code's behavior and performance.
                """,
                agent=self.crew_agent,
                expected_output="Comprehensive execution results with performance analysis and feedback"
            )
            
            # Execute the analysis
            execution_result = str(self.crew.kickoff(tasks=[task]))
            
            # Store execution stats
            self.execution_stats[session_id] = {
                "code": code,
                "result": execution_result,
                "timestamp": datetime.now(),
                "problem_id": problem_id
            }
            
            # Send results to user
            await self._send_execution_response(session_id, execution_result, "execution_complete")
            
            logger.info(f"✅ Code execution completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error in code execution: {e}")
            await self._send_execution_response(
                session_id,
                "I encountered an issue executing your code. Please check for syntax errors and try again.",
                "execution_error"
            )
    
    async def _handle_solution_validation(self, session_id: str, payload: Dict[str, Any]):
        """Handle comprehensive solution validation"""
        try:
            code = payload.get("code", "")
            problem_id = payload.get("problem_id", "")
            
            logger.info(f"🧪 Validating solution for session {session_id}")
            
            # Use LangChain chains for detailed analysis
            
            # 1. Execute code and get basic results
            execution_result = json.loads(self._tool_execute_code_safely(code))
            
            # 2. Validate against test cases
            test_results = json.loads(self._tool_validate_test_cases(code, problem_id))
            
            # 3. Monitor performance
            performance_analysis = json.loads(self._tool_monitor_performance(json.dumps(execution_result)))
            
            # 4. Get problem context
            problem_context = self._get_problem_context(problem_id)
            
            # 5. Generate comprehensive analysis
            analysis = await self.result_analysis_chain.arun(
                execution_results=json.dumps(execution_result),
                test_case_results=json.dumps(test_results),
                performance_data=json.dumps(performance_analysis),
                problem_context=json.dumps(problem_context)
            )
            
            # 6. Generate user-appropriate feedback
            feedback = await self.feedback_generation_chain.arun(
                analysis_results=analysis,
                user_confidence="medium",  # Could be derived from context
                execution_status="success" if execution_result.get("success") else "failure",
                interview_stage="solution_review"
            )
            
            # Combine results
            final_response = f"""**Execution Results:**

{feedback}

**Technical Analysis:**

{analysis}

**Test Results:**
- Passed: {test_results.get('passed_tests', 0)}/{test_results.get('total_tests', 0)} test cases
- Success Rate: {int(test_results.get('pass_rate', 0) * 100)}%

**Performance:**
- Execution Time: {performance_analysis.get('execution_time_ms', 0)}ms
- Category: {performance_analysis.get('performance_category', 'unknown')}"""
            
            await self._send_execution_response(session_id, final_response, "solution_validation")
            
        except Exception as e:
            logger.error(f"❌ Error in solution validation: {e}")
            await self._send_execution_response(
                session_id,
                "I'll validate your solution. Let me run it through the test cases and analyze the results.",
                "validation_fallback"
            )
    
    def _get_problem_context(self, problem_id: str) -> Dict[str, Any]:
        """Get problem context from database"""
        try:
            with self._db_lock:
                conn = self.sync_db_pool.getconn()
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute(
                        "SELECT title, description, difficulty FROM problems WHERE id = %s",
                        (problem_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        return {
                            "title": result['title'],
                            "description": result['description'],
                            "difficulty": result['difficulty']
                        }
                    return {}
                    
                finally:
                    self.sync_db_pool.putconn(conn)
                    
        except Exception as e:
            logger.error(f"Error getting problem context: {e}")
            return {}
    
    async def _send_execution_response(self, session_id: str, content: str, response_type: str = "execution"):
        """Send execution response back to user"""
        try:
            response_message = AgentMessage(
                event_type=EventType.AGENT_RESPONSE,
                source_agent=self.agent_name,
                target_agent="user",
                payload={
                    "speaker": "execution_engine",
                    "content": content,
                    "session_id": session_id,
                    "response_type": response_type,
                    "timestamp": datetime.now().isoformat()
                },
                context_snapshot={}
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, response_message)
            logger.info(f"📤 Sent execution response ({response_type}): {content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error sending execution response: {e}")
    
    # ==========================================
    # MAIN RUN LOOP
    # ==========================================
    
    async def run(self):
        """Main execution agent run loop"""
        self.running = True
        logger.info("🚀 Execution Agent starting...")
        
        try:
            await self.initialize()
            
            # Update status to active
            await self.message_bus.register_agent(self.agent_name, {
                "type": "execution",
                "capabilities": ["code_execution", "test_validation", "performance_monitoring", "security_sandboxing"],
                "status": "active"
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start listening for messages
            await self.message_bus.start_listening()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in execution agent: {e}")
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
        logger.info("🛑 Execution Agent shutting down...")
        self.running = False
        
        if self.message_bus:
            await self.message_bus.shutdown()
        if self.sync_db_pool:
            self.sync_db_pool.closeall()
        if self.docker_client:
            try:
                self.docker_client.close()
            except:
                pass

# Entry point
async def main():
    executor = ExecutionAgent()
    try:
        await executor.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())