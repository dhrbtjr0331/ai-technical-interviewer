package main

import (
	"encoding/json"
	"time"
)

// API Models
type StartInterviewRequest struct {
	UserID     string `json:"user_id" binding:"required"`
	Difficulty string `json:"difficulty"`
}

type StartInterviewResponse struct {
	SessionID  string `json:"session_id"`
	UserID     string `json:"user_id"`
	Difficulty string `json:"difficulty"`
	Status     string `json:"status"`
}

type SendMessageRequest struct {
	Content string `json:"content" binding:"required"`
}

type SendMessageResponse struct {
	MessageID string `json:"message_id"`
	Status    string `json:"status"`
}

type SubmitCodeRequest struct {
	Code     string `json:"code" binding:"required"`
	Language string `json:"language"`
}

type SubmitCodeResponse struct {
	Analysis string `json:"analysis"`
	Status   string `json:"status"`
}

type ExecuteCodeResponse struct {
	Output    string                 `json:"output"`
	Success   bool                   `json:"success"`
	Error     string                 `json:"error,omitempty"`
	TestCases []TestCaseResult       `json:"test_cases,omitempty"`
	Metrics   map[string]interface{} `json:"metrics,omitempty"`
}

type TestCaseResult struct {
	Input    interface{} `json:"input"`
	Expected interface{} `json:"expected"`
	Actual   interface{} `json:"actual"`
	Passed   bool        `json:"passed"`
}

type InterviewStatus struct {
	SessionID         string                 `json:"session_id"`
	UserID            string                 `json:"user_id"`
	Status            string                 `json:"status"`
	CurrentProblem    *Problem               `json:"current_problem,omitempty"`
	PerformanceMetrics map[string]interface{} `json:"performance_metrics"`
	Messages          []ChatMessage          `json:"messages"`
}

type Problem struct {
	ID          string      `json:"id"`
	Title       string      `json:"title"`
	Description string      `json:"description"`
	Difficulty  string      `json:"difficulty"`
	StarterCode string      `json:"starter_code"`
	Hints       []string    `json:"hints"`
	Topics      []string    `json:"topics"`
	TestCases   []TestCase  `json:"test_cases"`
}

type TestCase struct {
	InputData      interface{} `json:"input_data"`
	ExpectedOutput interface{} `json:"expected_output"`
	Description    string      `json:"description"`
	IsHidden       bool        `json:"is_hidden"`
}

type ChatMessage struct {
	ID        string    `json:"id"`
	Speaker   string    `json:"speaker"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Type      string    `json:"type"`
}

// Agent Message Models (matching Python models)
type AgentMessage struct {
	EventType       string                 `json:"event_type"`
	SourceAgent     string                 `json:"source_agent"`
	TargetAgent     string                 `json:"target_agent"`
	Payload         map[string]interface{} `json:"payload"`
	ContextSnapshot map[string]interface{} `json:"context_snapshot"`
	Timestamp       FlexTime               `json:"timestamp"`
	CorrelationID   string                 `json:"correlation_id"`
}

// FlexTime handles multiple timestamp formats from Python agents
type FlexTime struct {
	time.Time
}

func (ft *FlexTime) UnmarshalJSON(b []byte) error {
	s := string(b[1 : len(b)-1]) // Remove quotes
	
	// Try multiple timestamp formats
	formats := []string{
		"2006-01-02T15:04:05.000000",     // Python datetime without timezone
		"2006-01-02T15:04:05Z07:00",      // RFC3339 with timezone
		"2006-01-02T15:04:05.000000Z",    // Python datetime with Z
		time.RFC3339,                     // Standard RFC3339
		time.RFC3339Nano,                 // RFC3339 with nanoseconds
	}
	
	for _, format := range formats {
		if t, err := time.Parse(format, s); err == nil {
			ft.Time = t
			return nil
		}
	}
	
	return nil // Return nil to not break parsing, use zero time
}

func (a *AgentMessage) ToJSON() (string, error) {
	data, err := json.Marshal(a)
	return string(data), err
}

func AgentMessageFromJSON(jsonStr string) (*AgentMessage, error) {
	var msg AgentMessage
	err := json.Unmarshal([]byte(jsonStr), &msg)
	return &msg, err
}

// WebSocket Message Types
type WebSocketMessage struct {
	Type      string      `json:"type"`
	SessionID string      `json:"session_id,omitempty"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

const (
	// Event Types
	EventUserInput              = "user_input"
	EventCodeChange             = "code_change"
	EventExecutionRequest       = "execution_request"
	EventAgentResponse          = "agent_response"
	EventContextUpdate          = "context_update"
	EventInterviewStateChange   = "interview_state_change"
	EventProblemSelected        = "problem_selected"
	EventHintRequest            = "hint_request"
	EventEvaluationComplete     = "evaluation_complete"

	// Channels (matching Python channels)
	ChannelCoordination    = "coordination"
	ChannelUserInteraction = "user_interaction"
	ChannelCodeAnalysis    = "code_analysis"
	ChannelExecution       = "execution"
	ChannelEvaluation      = "evaluation"
	ChannelSystem          = "system"

	// WebSocket Message Types
	WSMessageChatMessage    = "chat_message"
	WSMessageCodeAnalysis   = "code_analysis"
	WSMessageExecutionResult = "execution_result"
	WSMessageStatusUpdate   = "status_update"
	WSMessageError          = "error"
)