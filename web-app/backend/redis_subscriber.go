package main

import (
	"context"
	"log"
	"time"

	"github.com/go-redis/redis/v8"
)

type RedisSubscriber struct {
	redis       *RedisClient
	agentService *AgentService
	wsHub       *WebSocketHub
	ctx         context.Context
	cancel      context.CancelFunc
}

func NewRedisSubscriber(redis *RedisClient, agentService *AgentService, wsHub *WebSocketHub) *RedisSubscriber {
	ctx, cancel := context.WithCancel(context.Background())
	return &RedisSubscriber{
		redis:       redis,
		agentService: agentService,
		wsHub:       wsHub,
		ctx:         ctx,
		cancel:      cancel,
	}
}

func (s *RedisSubscriber) Start() {
	// Subscribe to channels where agents send responses
	pubsub := s.redis.Subscribe(
		ChannelUserInteraction,
		ChannelCodeAnalysis,
		ChannelExecution,
		ChannelEvaluation,
		ChannelSystem,
	)

	defer pubsub.Close()

	log.Println("🔊 Redis subscriber started, listening for agent responses...")

	// Listen for messages
	ch := pubsub.Channel()
	for {
		select {
		case msg := <-ch:
			s.handleMessage(msg)
		case <-s.ctx.Done():
			log.Println("Redis subscriber stopped")
			return
		}
	}
}

func (s *RedisSubscriber) Stop() {
	s.cancel()
}

func (s *RedisSubscriber) handleMessage(msg *redis.Message) {
	// Parse agent message
	agentMsg, err := AgentMessageFromJSON(msg.Payload)
	if err != nil {
		log.Printf("Error parsing agent message: %v", err)
		return
	}

	// Only process messages targeting "user" or responses from agents
	if agentMsg.TargetAgent != "user" && agentMsg.EventType != EventAgentResponse {
		return
	}

	sessionID, ok := agentMsg.Payload["session_id"].(string)
	if !ok {
		log.Printf("No session_id in agent message")
		return
	}

	log.Printf("📨 Received agent response: %s -> %s (session: %s)", 
		agentMsg.SourceAgent, agentMsg.EventType, sessionID)

	// Update agent service with the response
	s.agentService.HandleAgentResponse(agentMsg)

	// Send real-time update via WebSocket
	s.sendWebSocketUpdate(sessionID, agentMsg)
}

func (s *RedisSubscriber) sendWebSocketUpdate(sessionID string, agentMsg *AgentMessage) {
	var wsMessage *WebSocketMessage

	switch agentMsg.EventType {
	case EventAgentResponse:
		content, _ := agentMsg.Payload["content"].(string)
		speaker, _ := agentMsg.Payload["speaker"].(string)
		responseType, _ := agentMsg.Payload["response_type"].(string)

		if content != "" && speaker != "" {
			wsMessage = &WebSocketMessage{
				Type:      WSMessageChatMessage,
				SessionID: sessionID,
				Data: map[string]interface{}{
					"id":        agentMsg.CorrelationID,
					"speaker":   speaker,
					"content":   content,
					"type":      responseType,
					"timestamp": agentMsg.Timestamp.Time,
				},
				Timestamp: time.Now(),
			}
		}

	case EventContextUpdate:
		wsMessage = &WebSocketMessage{
			Type:      WSMessageStatusUpdate,
			SessionID: sessionID,
			Data: map[string]interface{}{
				"context_update": agentMsg.Payload,
				"timestamp":      agentMsg.Timestamp.Time,
			},
			Timestamp: time.Now(),
		}

	case EventInterviewStateChange:
		status, _ := agentMsg.Payload["status"].(string)
		wsMessage = &WebSocketMessage{
			Type:      WSMessageStatusUpdate,
			SessionID: sessionID,
			Data: map[string]interface{}{
				"status":    status,
				"timestamp": agentMsg.Timestamp.Time,
			},
			Timestamp: time.Now(),
		}

	default:
		// Generic update for other event types
		wsMessage = &WebSocketMessage{
			Type:      WSMessageStatusUpdate,
			SessionID: sessionID,
			Data: map[string]interface{}{
				"event_type": agentMsg.EventType,
				"payload":    agentMsg.Payload,
				"timestamp":  agentMsg.Timestamp.Time,
			},
			Timestamp: time.Now(),
		}
	}

	if wsMessage != nil {
		s.wsHub.BroadcastToSession(sessionID, wsMessage)
	}
}