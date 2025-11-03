package main

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// LoadTestConfig holds configuration for load testing
type LoadTestConfig struct {
	NumSessions      int
	BaseURL          string
	MessagesPerSession int
	RampUpTime       time.Duration
}

// LoadTestResults holds the results of a load test
type LoadTestResults struct {
	TotalSessions    int
	SuccessfulConns  int
	FailedConns      int
	TotalMessages    int
	AvgResponseTime  time.Duration
	MaxResponseTime  time.Duration
	MinResponseTime  time.Duration
	Errors           []string
	Duration         time.Duration
}

// LoadTester performs concurrent WebSocket connection tests
type LoadTester struct {
	config  LoadTestConfig
	results LoadTestResults
	mu      sync.Mutex
}

// NewLoadTester creates a new load tester
func NewLoadTester(config LoadTestConfig) *LoadTester {
	return &LoadTester{
		config: config,
		results: LoadTestResults{
			TotalSessions: config.NumSessions,
			MinResponseTime: time.Hour, // Start with a large value
		},
	}
}

// RunLoadTest executes the load test
func (lt *LoadTester) RunLoadTest() LoadTestResults {
	startTime := time.Now()
	var wg sync.WaitGroup

	// Calculate delay between connection starts for ramp-up
	rampUpDelay := lt.config.RampUpTime / time.Duration(lt.config.NumSessions)

	for i := 0; i < lt.config.NumSessions; i++ {
		wg.Add(1)
		go lt.simulateSession(i, &wg)

		// Ramp-up: add delay between starting connections
		if i < lt.config.NumSessions-1 {
			time.Sleep(rampUpDelay)
		}
	}

	wg.Wait()
	lt.results.Duration = time.Since(startTime)

	return lt.results
}

// simulateSession simulates a single interview session with WebSocket
func (lt *LoadTester) simulateSession(sessionID int, wg *sync.WaitGroup) {
	defer wg.Done()

	wsURL := fmt.Sprintf("ws://localhost:8081/ws?session_id=load-test-%d", sessionID)

	// Connect to WebSocket
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		lt.recordError(fmt.Sprintf("Session %d: Failed to connect: %v", sessionID, err))
		lt.incrementFailedConns()
		return
	}
	defer conn.Close()

	lt.incrementSuccessfulConns()

	// Send messages and measure response times
	for msgNum := 0; msgNum < lt.config.MessagesPerSession; msgNum++ {
		message := fmt.Sprintf("Test message %d from session %d", msgNum, sessionID)

		sendStart := time.Now()

		// Send message
		err = conn.WriteMessage(websocket.TextMessage, []byte(message))
		if err != nil {
			lt.recordError(fmt.Sprintf("Session %d: Failed to send message: %v", sessionID, err))
			continue
		}

		// Wait for response
		_, response, err := conn.ReadMessage()
		if err != nil {
			lt.recordError(fmt.Sprintf("Session %d: Failed to read response: %v", sessionID, err))
			continue
		}

		responseTime := time.Since(sendStart)
		lt.recordMessage(responseTime)

		// Log response for debugging (only for first session)
		if sessionID == 0 && msgNum == 0 {
			log.Printf("Session %d received response: %s (took %v)", sessionID, string(response), responseTime)
		}

		// Small delay between messages within a session
		time.Sleep(100 * time.Millisecond)
	}

	log.Printf("Session %d completed successfully", sessionID)
}

// Record statistics methods
func (lt *LoadTester) incrementSuccessfulConns() {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.results.SuccessfulConns++
}

func (lt *LoadTester) incrementFailedConns() {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.results.FailedConns++
}

func (lt *LoadTester) recordMessage(responseTime time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()

	lt.results.TotalMessages++

	// Update average
	if lt.results.TotalMessages == 1 {
		lt.results.AvgResponseTime = responseTime
	} else {
		// Calculate running average
		total := lt.results.AvgResponseTime * time.Duration(lt.results.TotalMessages-1)
		lt.results.AvgResponseTime = (total + responseTime) / time.Duration(lt.results.TotalMessages)
	}

	// Update max
	if responseTime > lt.results.MaxResponseTime {
		lt.results.MaxResponseTime = responseTime
	}

	// Update min
	if responseTime < lt.results.MinResponseTime {
		lt.results.MinResponseTime = responseTime
	}
}

func (lt *LoadTester) recordError(errMsg string) {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.results.Errors = append(lt.results.Errors, errMsg)
}

// PrintResults prints the load test results in a formatted way
func (lt *LoadTester) PrintResults() {
	fmt.Println("\n" + string([]rune(string("="*80))))
	fmt.Println("LOAD TEST RESULTS")
	fmt.Println(string([]rune(string("="*80))))
	fmt.Printf("Total Sessions:       %d\n", lt.results.TotalSessions)
	fmt.Printf("Successful Connections: %d\n", lt.results.SuccessfulConns)
	fmt.Printf("Failed Connections:   %d\n", lt.results.FailedConns)
	fmt.Printf("Connection Success Rate: %.2f%%\n", float64(lt.results.SuccessfulConns)/float64(lt.results.TotalSessions)*100)
	fmt.Println(string([]rune(string("-"*80))))
	fmt.Printf("Total Messages Sent:  %d\n", lt.results.TotalMessages)
	fmt.Printf("Avg Response Time:    %v\n", lt.results.AvgResponseTime)
	fmt.Printf("Min Response Time:    %v\n", lt.results.MinResponseTime)
	fmt.Printf("Max Response Time:    %v\n", lt.results.MaxResponseTime)
	fmt.Println(string([]rune(string("-"*80))))
	fmt.Printf("Test Duration:        %v\n", lt.results.Duration)
	fmt.Printf("Throughput:           %.2f messages/sec\n", float64(lt.results.TotalMessages)/lt.results.Duration.Seconds())
	fmt.Println(string([]rune(string("-"*80))))

	if len(lt.results.Errors) > 0 {
		fmt.Printf("Errors: %d\n", len(lt.results.Errors))
		fmt.Println("First 5 errors:")
		for i, err := range lt.results.Errors {
			if i >= 5 {
				break
			}
			fmt.Printf("  - %s\n", err)
		}
	} else {
		fmt.Println("No errors!")
	}
	fmt.Println(string([]rune(string("="*80))))
}

// RunLoadTestsuite runs multiple test scenarios
func RunLoadTestSuite() {
	scenarios := []LoadTestConfig{
		{
			NumSessions:        10,
			BaseURL:            "localhost:8081",
			MessagesPerSession: 3,
			RampUpTime:         2 * time.Second,
		},
		{
			NumSessions:        25,
			BaseURL:            "localhost:8081",
			MessagesPerSession: 5,
			RampUpTime:         5 * time.Second,
		},
		{
			NumSessions:        50,
			BaseURL:            "localhost:8081",
			MessagesPerSession: 5,
			RampUpTime:         10 * time.Second,
		},
		{
			NumSessions:        100,
			BaseURL:            "localhost:8081",
			MessagesPerSession: 3,
			RampUpTime:         15 * time.Second,
		},
	}

	for i, scenario := range scenarios {
		fmt.Printf("\n\n########## Running Scenario %d: %d Concurrent Sessions ##########\n", i+1, scenario.NumSessions)
		tester := NewLoadTester(scenario)
		tester.RunLoadTest()
		tester.PrintResults()

		// Cool-down period between scenarios
		if i < len(scenarios)-1 {
			fmt.Println("\nCooling down for 5 seconds...")
			time.Sleep(5 * time.Second)
		}
	}

	fmt.Println("\n\n✅ Load test suite completed!")
}

func main() {
	log.Println("Starting AI Interviewer Load Testing Suite...")
	log.Println("Make sure the backend server is running on localhost:8081")
	log.Println("")

	time.Sleep(2 * time.Second)

	RunLoadTestSuite()
}
