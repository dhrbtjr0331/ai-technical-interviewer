package main

import (
	"log"
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize Redis client
	redisClient := NewRedisClient("localhost:6379")
	defer redisClient.Close()

	// Initialize agent service
	agentService := NewAgentService(redisClient)

	// Initialize WebSocket hub
	wsHub := NewWebSocketHub()
	go wsHub.Run()

	// Initialize Redis subscriber for agent responses
	redisSubscriber := NewRedisSubscriber(redisClient, agentService, wsHub)
	go redisSubscriber.Start()
	defer redisSubscriber.Stop()

	// Setup Gin router
	r := gin.Default()

	// CORS middleware
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"}, // React dev server
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		AllowCredentials: true,
	}))

	// API routes
	api := r.Group("/api/v1")
	{
		// Interview management
		api.POST("/interviews", func(c *gin.Context) {
			handleStartInterview(c, agentService)
		})
		api.GET("/interviews/:sessionId", func(c *gin.Context) {
			handleGetInterviewStatus(c, agentService)
		})
		api.DELETE("/interviews/:sessionId", func(c *gin.Context) {
			handleEndInterview(c, agentService)
		})

		// Chat/messaging
		api.POST("/interviews/:sessionId/messages", func(c *gin.Context) {
			handleSendMessage(c, agentService)
		})

		// Code submission
		api.POST("/interviews/:sessionId/code", func(c *gin.Context) {
			handleSubmitCode(c, agentService)
		})

		// Code execution
		api.POST("/interviews/:sessionId/execute", func(c *gin.Context) {
			handleExecuteCode(c, agentService)
		})

		// Problems
		api.GET("/problems", func(c *gin.Context) {
			handleGetProblems(c, agentService)
		})
	}

	// WebSocket endpoint
	r.GET("/ws/:sessionId", func(c *gin.Context) {
		handleWebSocket(c, wsHub)
	})

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok", "service": "ai-interviewer-bff"})
	})

	log.Println("🚀 AI Interviewer BFF starting on :8081")
	log.Fatal(r.Run(":8081"))
}