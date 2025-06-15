from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

class EventType(Enum):
    USER_INPUT = "user_input"
    CODE_CHANGE = "code_change"
    EXECUTION_REQUEST = "execution_request"
    AGENT_RESPONSE = "agent_response"
    CONTEXT_UPDATE = "context_update"
    INTERVIEW_STATE_CHANGE = "interview_state_change"
    PROBLEM_SELECTED = "problem_selected"
    HINT_REQUEST = "hint_request"
    EVALUATION_COMPLETE = "evaluation_complete"

class InterviewState(Enum):
    INITIALIZING = "initializing"
    PROBLEM_INTRODUCTION = "problem_intro"
    CLARIFICATION = "clarification"
    ACTIVE_CODING = "active_coding"
    DEBUGGING = "debugging"
    EXECUTION_REVIEW = "execution_review"
    SOLUTION_DISCUSSION = "solution_discussion"
    OPTIMIZATION = "optimization"
    COMPLETED = "completed"

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class TestCase:
    input_data: Any
    expected_output: Any
    description: str = ""
    is_hidden: bool = False

@dataclass
class Problem:
    id: str
    title: str
    description: str
    difficulty: Difficulty
    test_cases: List[TestCase]
    starter_code: str = ""
    hints: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    time_limit_seconds: int = 3600  # 1 hour default
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "test_cases": [
                {
                    "input_data": tc.input_data,
                    "expected_output": tc.expected_output,
                    "description": tc.description,
                    "is_hidden": tc.is_hidden
                } for tc in self.test_cases
            ],
            "starter_code": self.starter_code,
            "hints": self.hints,
            "topics": self.topics,
            "time_limit_seconds": self.time_limit_seconds
        }

@dataclass
class CodeSnapshot:
    code: str
    timestamp: datetime
    is_valid_syntax: bool
    line_count: int
    approach_detected: Optional[str] = None

@dataclass
class ExecutionResult:
    success: bool
    output: str
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    test_case_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    start_time: datetime
    current_duration_seconds: int = 0
    code_changes: int = 0
    execution_attempts: int = 0
    hints_used: int = 0
    questions_asked: int = 0

@dataclass
class Message:
    speaker: str  # "user" or agent name
    content: str
    timestamp: datetime
    message_type: str = "text"  # "text", "code", "system"

@dataclass
class InterviewContext:
    session_id: str
    user_id: str
    current_problem: Optional[Problem] = None
    current_code: str = ""
    code_history: List[CodeSnapshot] = field(default_factory=list)
    conversation_history: List[Message] = field(default_factory=list)
    interview_state: InterviewState = InterviewState.INITIALIZING
    performance_metrics: PerformanceMetrics = field(default_factory=lambda: PerformanceMetrics(start_time=datetime.now()))
    agent_memory: Dict[str, Any] = field(default_factory=dict)
    last_execution_result: Optional[ExecutionResult] = None
    
    def update_code(self, new_code: str, is_valid_syntax: bool = True, approach_detected: str = None):
        """Thread-safe code updates"""
        self.current_code = new_code
        snapshot = CodeSnapshot(
            code=new_code,
            timestamp=datetime.now(),
            is_valid_syntax=is_valid_syntax,
            line_count=len(new_code.split('\n')),
            approach_detected=approach_detected
        )
        self.code_history.append(snapshot)
        self.performance_metrics.code_changes += 1
    
    def add_conversation(self, speaker: str, content: str, message_type: str = "text"):
        """Add to conversation with context awareness"""
        message = Message(
            speaker=speaker,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type
        )
        self.conversation_history.append(message)
        
        if speaker == "user":
            self.performance_metrics.questions_asked += 1
    
    def get_agent_context(self, agent_name: str) -> Dict[str, Any]:
        """Get agent-specific context"""
        return self.agent_memory.get(agent_name, {})
    
    def update_agent_context(self, agent_name: str, context_update: Dict[str, Any]):
        """Update agent-specific context"""
        if agent_name not in self.agent_memory:
            self.agent_memory[agent_name] = {}
        self.agent_memory[agent_name].update(context_update)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "current_problem": self.current_problem.to_dict() if self.current_problem else None,
            "current_code": self.current_code,
            "interview_state": self.interview_state.value,
            "performance_metrics": {
                "start_time": self.performance_metrics.start_time.isoformat(),
                "current_duration_seconds": self.performance_metrics.current_duration_seconds,
                "code_changes": self.performance_metrics.code_changes,
                "execution_attempts": self.performance_metrics.execution_attempts,
                "hints_used": self.performance_metrics.hints_used,
                "questions_asked": self.performance_metrics.questions_asked
            },
            "conversation_count": len(self.conversation_history),
            "code_history_count": len(self.code_history)
        }

@dataclass
class AgentMessage:
    event_type: EventType
    source_agent: str
    target_agent: str  # or "broadcast" for all agents
    payload: Dict[str, Any]
    context_snapshot: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_json(self) -> str:
        """Serialize for Redis transmission"""
        return json.dumps({
            "event_type": self.event_type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "payload": self.payload,
            "context_snapshot": self.context_snapshot,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """Deserialize from Redis"""
        data = json.loads(json_str)
        return cls(
            event_type=EventType(data["event_type"]),
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            payload=data["payload"],
            context_snapshot=data["context_snapshot"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"]
        )