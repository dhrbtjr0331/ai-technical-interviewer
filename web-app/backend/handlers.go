package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func handleStartInterview(c *gin.Context, agentService *AgentService) {
	var req StartInterviewRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set default difficulty if not provided
	if req.Difficulty == "" {
		req.Difficulty = "medium"
	}

	response, err := agentService.StartInterview(req.UserID, req.Difficulty)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, response)
}

func handleGetInterviewStatus(c *gin.Context, agentService *AgentService) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	status, err := agentService.GetInterviewStatus(sessionID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, status)
}

func handleEndInterview(c *gin.Context, agentService *AgentService) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	err := agentService.EndInterview(sessionID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ended"})
}

func handleSendMessage(c *gin.Context, agentService *AgentService) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	var req SendMessageRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	response, err := agentService.SendMessage(sessionID, req.Content)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func handleSubmitCode(c *gin.Context, agentService *AgentService) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	var req SubmitCodeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set default language if not provided
	if req.Language == "" {
		req.Language = "python"
	}

	response, err := agentService.SubmitCode(sessionID, req.Code, req.Language)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func handleExecuteCode(c *gin.Context, agentService *AgentService) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	response, err := agentService.ExecuteCode(sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func handleGetProblems(c *gin.Context, agentService *AgentService) {
	// For now, return mock problems
	// In the future, this could query the AI service for available problems
	problems := []Problem{
		{
			ID:          "two_sum",
			Title:       "Two Sum",
			Description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
			Difficulty:  "easy",
			StarterCode: "def two_sum(nums, target):\n    # Your solution here\n    pass",
			Hints:       []string{"Try using a hash map", "You can solve this in O(n) time"},
			Topics:      []string{"Array", "Hash Table"},
			TestCases: []TestCase{
				{
					InputData:      map[string]interface{}{"nums": []int{2, 7, 11, 15}, "target": 9},
					ExpectedOutput: []int{0, 1},
					Description:    "Basic case with target at beginning",
					IsHidden:       false,
				},
			},
		},
		{
			ID:          "valid_parentheses",
			Title:       "Valid Parentheses",
			Description: "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
			Difficulty:  "easy",
			StarterCode: "def is_valid(s):\n    # Your solution here\n    pass",
			Hints:       []string{"Stack data structure is perfect for this", "Push opening brackets, pop when you see closing ones"},
			Topics:      []string{"String", "Stack"},
			TestCases: []TestCase{
				{
					InputData:      map[string]interface{}{"s": "()"},
					ExpectedOutput: true,
					Description:    "Valid parentheses",
					IsHidden:       false,
				},
			},
		},
	}

	c.JSON(http.StatusOK, gin.H{"problems": problems})
}