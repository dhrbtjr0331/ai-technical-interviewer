import asyncio
import os
import logging
from datetime import datetime
from typing import Dict, Any

# We'll use direct Anthropic client for now, but structure it for easy LangChain migration
from anthropic import AsyncAnthropic

from shared.message_bus import MessageBus, Channels, get_message_bus
from shared.models import AgentMessage, EventType, InterviewContext

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class HelloWorldCoordinator:
    """A simple test coordinator that manages the HelloWorld agent."""

    def __init__(self):
        self.agent_name = "hello_world"
        self.claude_client: AsyncAnthropic = None
        self.message_bus: MessageBus = None
        self.running = False

    async def initialize(self):
        """Initialize the coordinator."""
        try:
            # Initialize Claude client
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key or api_key == "YOUR_CLAUDE_API_KEY":
                raise ValueError("CLAUDE_API_KEY environment variable is not set")
            
            self.claude_client = AsyncAnthropic(api_key=api_key)

            # Initialize the message bus
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.message_bus = await get_message_bus(redis_url)

            # Register agent in discovery service
            await self.message_bus.register_agent(self.agent_name, {
                "type": "test_agent",
                "capabilities": ["chat", "test_responses"],
                "status": "initializing"
            })

            # Subscribe to channels we care about
            self.message_bus.subscribe(Channels.USER_INTERACTION, self.handle_user_message)
            self.message_bus.subscribe(Channels.SYSTEM, self.handle_system_message)

            logger.info(f"✅ {self.agent_name} initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.agent_name}: {e}")
            raise
    
    async def handle_user_message(self, message: AgentMessage):
        """Handle user interaction messages"""
        try:
            payload = message.payload

            # Only respond to messages meant for us or broadcast
            if message.target_agent not in [self.agent_name, "broadcast", "any"]:
                return

            user_content = payload.get("content", "")
            session_id = payload.get("session_id", "unknown")

            logger.info(f"🔍 Received user message for session {session_id}: {user_content}")

            # Generate a response using Claude
            response = await self.generate_claude_response(user_content, payload)

            # Send response back to the user through message bus
            await self.send_response(session_id, response, message.correlation_id)
        
        except Exception as e:
            logger.error(f"❌ Error handling user message: {e}")
    
    async def handle_system_message(self, message: AgentMessage):
        """Handle system-level message"""
        try:
            payload = message.payload
            event_type = payload.get("event", "unknown")

            if event_type == "health_check":
                await self.send_heartbeat()
            elif event_type == "shutdown":
                await self.shutdown()
            
            logger.info(f"🔧 System event: {event_type}")
        
        except Exception as e:
            logger.error(f"❌ Error handling system message: {e}")
    
    async def generate_claude_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        Generate a response using Claude
        NOTE: This will be replaced with LangChain + CrewAI in Phase 1B
        """
        try:
            # Prepare the prompt
            system_prompt = """You are a friendly test agent for an AI technical interview system. 
            You're helping verify that the message bus and Claude integration work correctly.
            
            Respond helpfully but mention that you're a test agent. Keep responses concise but friendly.
            If the user asks about coding problems, mention that the full interview system is still being built.""" 

            # Call Claude
            response = await self.claude_client.messages.create(
                model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
                max_tokens=500,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": user_input
                }]
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def send_response(self, session_id: str, content: str, correlation_id: str):
        """Send response back through message bus"""
        try:
            response_message = AgentMessage(
                event_type=EventType.AGENT_RESPONSE,
                source_agent=self.agent_name,
                target_agent="user",
                payload={
                    "speaker": "hello_world_agent",
                    "content": content,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                },
                context_snapshot={},
                correlation_id=correlation_id
            )
            
            await self.message_bus.publish(Channels.USER_INTERACTION, response_message)
            logger.info(f"📤 Sent response: {content[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error sending response: {e}")
    
    async def send_heartbeat(self):
        """Send heartbeat to keep agent registered as active"""
        try:
            await self.message_bus.heartbeat(self.agent_name)
            logger.debug(f"💓 Heartbeat sent")
        except Exception as e:
            logger.error(f"❌ Heartbeat error: {e}")
    
    async def run(self):
        """Main agent run loop"""
        self.running = True
        logger.info(f"🚀 {self.agent_name} starting...")
        
        try:
            await self.initialize()
            
            # Update status to active
            await self.message_bus.register_agent(self.agent_name, {
                "type": "test_agent",
                "capabilities": ["chat", "test_responses"],
                "status": "active"
            })
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            
            # Start listening for messages
            await self.message_bus.start_listening()
            
        except Exception as e:
            logger.error(f"❌ Fatal error in {self.agent_name}: {e}")
        finally:
            await self.shutdown()
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                await self.send_heartbeat()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"❌ Heartbeat loop error: {e}")
                break
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info(f"🛑 {self.agent_name} shutting down...")
        self.running = False
        
        if self.message_bus:
            await self.message_bus.shutdown()
    
# Entry point
async def main():
    agent = HelloWorldAgent()
    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())