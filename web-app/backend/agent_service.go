package main

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
)

type AgentService struct {
	redis   *RedisClient
	sessions map[string]*InterviewStatus
}

func NewAgentService(redis *RedisClient) *AgentService {
	return &AgentService{
		redis:   redis,
		sessions: make(map[string]*InterviewStatus),
	}
}

func (s *AgentService) StartInterview(userID, difficulty string) (*StartInterviewResponse, error) {
	sessionID := uuid.New().String()

	// Create agent message to start interview
	agentMsg := &AgentMessage{
		EventType:       EventInterviewStateChange,
		SourceAgent:     "web_bff",
		TargetAgent:     "coordinator",
		Payload: map[string]interface{}{
			"action":      "start_interview",
			"session_id":  sessionID,
			"user_id":     userID,
			"difficulty":  difficulty,
		},
		ContextSnapshot: map[string]interface{}{},
		Timestamp:       FlexTime{time.Now()},
		CorrelationID:   uuid.New().String(),
	}

	// Send message to coordinator
	msgJSON, err := agentMsg.ToJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize message: %w", err)
	}

	err = s.redis.Publish(ChannelCoordination, msgJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to publish message: %w", err)
	}

	// Store session info
	session := &InterviewStatus{
		SessionID:          sessionID,
		UserID:             userID,
		Status:             "starting",
		PerformanceMetrics: make(map[string]interface{}),
		Messages:           []ChatMessage{},
	}
	s.sessions[sessionID] = session

	log.Printf("Started interview session %s for user %s", sessionID, userID)

	return &StartInterviewResponse{
		SessionID:  sessionID,
		UserID:     userID,
		Difficulty: difficulty,
		Status:     "started",
	}, nil
}

func (s *AgentService) SendMessage(sessionID, content string) (*SendMessageResponse, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	messageID := uuid.New().String()

	// Add message to session
	message := ChatMessage{
		ID:        messageID,
		Speaker:   "user",
		Content:   content,
		Timestamp: time.Now(),
		Type:      "text",
	}
	session.Messages = append(session.Messages, message)

	// Create agent message
	agentMsg := &AgentMessage{
		EventType:   EventUserInput,
		SourceAgent: "web_bff",
		TargetAgent: "coordinator",
		Payload: map[string]interface{}{
			"session_id": sessionID,
			"content":    content,
			"user_id":    session.UserID,
		},
		ContextSnapshot: map[string]interface{}{
			"session_id": sessionID,
		},
		Timestamp:     FlexTime{time.Now()},
		CorrelationID: messageID,
	}

	// Send to coordinator for routing
	msgJSON, err := agentMsg.ToJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize message: %w", err)
	}

	err = s.redis.Publish(ChannelUserInteraction, msgJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to publish message: %w", err)
	}

	log.Printf("Sent message from session %s: %s", sessionID, content)

	return &SendMessageResponse{
		MessageID: messageID,
		Status:    "sent",
	}, nil
}

func (s *AgentService) SubmitCode(sessionID, code, language string) (*SubmitCodeResponse, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	// Create agent message for coordinator to route
	agentMsg := &AgentMessage{
		EventType:   EventCodeChange,
		SourceAgent: "web_bff",
		TargetAgent: "coordinator",
		Payload: map[string]interface{}{
			"action":     "submit_code",
			"session_id": sessionID,
			"code":       code,
			"language":   language,
			"user_id":    session.UserID,
		},
		ContextSnapshot: map[string]interface{}{
			"session_id": sessionID,
		},
		Timestamp:     FlexTime{time.Now()},
		CorrelationID: uuid.New().String(),
	}

	// Send to coordinator for routing to code analyzer
	msgJSON, err := agentMsg.ToJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize message: %w", err)
	}

	err = s.redis.Publish(ChannelCoordination, msgJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to publish message: %w", err)
	}

	log.Printf("Submitted code for session %s", sessionID)

	return &SubmitCodeResponse{
		Status: "submitted",
	}, nil
}

func (s *AgentService) ExecuteCode(sessionID string) (*ExecuteCodeResponse, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	// Create agent message for coordinator to route
	agentMsg := &AgentMessage{
		EventType:   EventExecutionRequest,
		SourceAgent: "web_bff",
		TargetAgent: "coordinator",
		Payload: map[string]interface{}{
			"action":     "execute_code",
			"session_id": sessionID,
			"user_id":    session.UserID,
		},
		ContextSnapshot: map[string]interface{}{
			"session_id": sessionID,
		},
		Timestamp:     FlexTime{time.Now()},
		CorrelationID: uuid.New().String(),
	}

	// Send to coordinator for routing to execution agent
	msgJSON, err := agentMsg.ToJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize message: %w", err)
	}

	err = s.redis.Publish(ChannelCoordination, msgJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to publish message: %w", err)
	}

	log.Printf("Requested code execution for session %s", sessionID)

	return &ExecuteCodeResponse{
		Success: true,
		Output:  "Execution request sent to agent",
	}, nil
}

func (s *AgentService) GetInterviewStatus(sessionID string) (*InterviewStatus, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	return session, nil
}

func (s *AgentService) EndInterview(sessionID string) error {
	session, exists := s.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	// Create agent message to end interview
	agentMsg := &AgentMessage{
		EventType:   EventInterviewStateChange,
		SourceAgent: "web_bff",
		TargetAgent: "coordinator",
		Payload: map[string]interface{}{
			"action":     "end_interview",
			"session_id": sessionID,
			"user_id":    session.UserID,
		},
		ContextSnapshot: map[string]interface{}{
			"session_id": sessionID,
		},
		Timestamp:     FlexTime{time.Now()},
		CorrelationID: uuid.New().String(),
	}

	// Send to coordinator
	msgJSON, err := agentMsg.ToJSON()
	if err != nil {
		return fmt.Errorf("failed to serialize message: %w", err)
	}

	err = s.redis.Publish(ChannelCoordination, msgJSON)
	if err != nil {
		return fmt.Errorf("failed to publish message: %w", err)
	}

	// Update session status
	session.Status = "ended"

	log.Printf("Ended interview session %s", sessionID)
	return nil
}

func (s *AgentService) HandleAgentResponse(message *AgentMessage) {
	sessionID, ok := message.Payload["session_id"].(string)
	if !ok {
		log.Printf("No session_id in agent response")
		return
	}

	session, exists := s.sessions[sessionID]
	if !exists {
		log.Printf("Session not found for response: %s", sessionID)
		return
	}

	// Handle different types of agent responses
	switch message.EventType {
	case EventAgentResponse:
		content, _ := message.Payload["content"].(string)
		speaker, _ := message.Payload["speaker"].(string)
		
		if content != "" && speaker != "" {
			chatMsg := ChatMessage{
				ID:        uuid.New().String(),
				Speaker:   speaker,
				Content:   content,
				Timestamp: time.Now(),
				Type:      "text",
			}
			session.Messages = append(session.Messages, chatMsg)
		}

	case EventContextUpdate:
		// Update session context
		if problem, ok := message.Payload["current_problem"]; ok {
			problemData, _ := json.Marshal(problem)
			var p Problem
			if err := json.Unmarshal(problemData, &p); err == nil {
				session.CurrentProblem = &p
			}
		}

	case EventInterviewStateChange:
		if status, ok := message.Payload["status"].(string); ok {
			session.Status = status
		}
	}

	log.Printf("Handled agent response for session %s: %s", sessionID, message.EventType)
}