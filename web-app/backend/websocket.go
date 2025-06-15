package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

type WebSocketHub struct {
	clients    map[string]map[*WebSocketClient]bool // sessionID -> clients
	register   chan *WebSocketClient
	unregister chan *WebSocketClient
	broadcast  chan *BroadcastMessage
}

type WebSocketClient struct {
	hub       *WebSocketHub
	conn      *websocket.Conn
	send      chan []byte
	sessionID string
}

type BroadcastMessage struct {
	SessionID string
	Message   *WebSocketMessage
}

func NewWebSocketHub() *WebSocketHub {
	return &WebSocketHub{
		clients:    make(map[string]map[*WebSocketClient]bool),
		register:   make(chan *WebSocketClient),
		unregister: make(chan *WebSocketClient),
		broadcast:  make(chan *BroadcastMessage),
	}
}

func (h *WebSocketHub) Run() {
	for {
		select {
		case client := <-h.register:
			if h.clients[client.sessionID] == nil {
				h.clients[client.sessionID] = make(map[*WebSocketClient]bool)
			}
			h.clients[client.sessionID][client] = true
			log.Printf("WebSocket client registered for session %s", client.sessionID)

		case client := <-h.unregister:
			if clients, ok := h.clients[client.sessionID]; ok {
				if _, ok := clients[client]; ok {
					delete(clients, client)
					close(client.send)
					if len(clients) == 0 {
						delete(h.clients, client.sessionID)
					}
					log.Printf("WebSocket client unregistered for session %s", client.sessionID)
				}
			}

		case broadcast := <-h.broadcast:
			if clients, ok := h.clients[broadcast.SessionID]; ok {
				messageBytes, err := json.Marshal(broadcast.Message)
				if err != nil {
					log.Printf("Error marshaling broadcast message: %v", err)
					continue
				}

				for client := range clients {
					select {
					case client.send <- messageBytes:
					default:
						delete(clients, client)
						close(client.send)
					}
				}

				if len(clients) == 0 {
					delete(h.clients, broadcast.SessionID)
				}
			}
		}
	}
}

func (h *WebSocketHub) BroadcastToSession(sessionID string, message *WebSocketMessage) {
	select {
	case h.broadcast <- &BroadcastMessage{
		SessionID: sessionID,
		Message:   message,
	}:
	default:
		log.Printf("Failed to broadcast to session %s - channel full", sessionID)
	}
}

func handleWebSocket(c *gin.Context, hub *WebSocketHub) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	client := &WebSocketClient{
		hub:       hub,
		conn:      conn,
		send:      make(chan []byte, 256),
		sessionID: sessionID,
	}

	client.hub.register <- client

	// Start goroutines for reading and writing
	go client.writePump()
	go client.readPump()
}

func (c *WebSocketClient) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		// Handle incoming WebSocket messages if needed
		log.Printf("Received WebSocket message from session %s: %s", c.sessionID, string(message))
	}
}

func (c *WebSocketClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to the current message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}