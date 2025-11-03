package graph

import (
	"sync"
	"time"
)

// This file will not be regenerated automatically.
//
// It serves as dependency injection for your app, add any dependencies you require here.

type Resolver struct {
	AgentService *AgentService
	Hub          *WebSocketHub

	// For subscriptions
	subscribers   map[string]map[chan interface{}]bool
	subscribersMu sync.RWMutex
}

// AgentService interface (imported from main package)
type AgentService interface {
	StartInterview(userID, difficulty string) (*StartInterviewResponse, error)
	SendMessage(sessionID, content string) (*SendMessageResponse, error)
	SubmitCode(sessionID, code, language string) (*SubmitCodeResponse, error)
	ExecuteCode(sessionID string) (*ExecuteCodeResponse, error)
	GetInterviewStatus(sessionID string) (*InterviewStatus, error)
	EndInterview(sessionID string) error
}

type WebSocketHub interface {
	// Add interface methods if needed for subscriptions
}

type StartInterviewResponse struct {
	SessionID  string
	UserID     string
	Difficulty string
	Status     string
}

type SendMessageResponse struct {
	MessageID string
	Status    string
}

type SubmitCodeResponse struct {
	Status string
}

type ExecuteCodeResponse struct {
	Success bool
	Output  string
}

type InterviewStatus struct {
	SessionID          string
	UserID             string
	Status             string
	PerformanceMetrics map[string]interface{}
	Messages           []ChatMessage
	CurrentProblem     *Problem
}

type ChatMessage struct {
	ID        string
	Speaker   string
	Content   string
	Timestamp time.Time
	Type      string
}

type Problem struct {
	ID          string
	Title       string
	Description string
	Difficulty  string
	Topics      []string
	StarterCode string
}

func NewResolver(agentService *AgentService, hub *WebSocketHub) *Resolver {
	return &Resolver{
		AgentService:  agentService,
		Hub:           hub,
		subscribers:   make(map[string]map[chan interface{}]bool),
		subscribersMu: sync.RWMutex{},
	}
}
