import asyncio
import json
import logging
from typing import Callable, Dict, Any, Optional, List
import redis.asyncio as redis
from contextlib import asynccontextmanager
from .models import AgentMessage, EventType, InterviewContext

logger = logging.getLogger(__name__)

class MessageBus:
    """Redis-based message bus for agent communication"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def initialize(self):
        """Initialize Redis connections"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            logger.info("Message bus initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize message bus: {e}")
            raise
    
    async def shutdown(self):
        """Clean shutdown of Redis connections"""
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Message bus shut down")
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for message bus lifecycle"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.shutdown()
    
    def subscribe(self, channel: str, callback: Callable[[AgentMessage], None]):
        """Subscribe to a channel with callback"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)
        logger.info(f"Subscribed to channel: {channel}")
    
    async def publish(self, channel: str, message: AgentMessage):
        """Publish message to channel"""
        try:
            await self.redis_client.publish(channel, message.to_json())
            
            # Also store in message history for debugging
            await self.redis_client.lpush(
                f"history:{channel}", 
                message.to_json()
            )
            # Keep only last 100 messages
            await self.redis_client.ltrim(f"history:{channel}", 0, 99)
            
            logger.debug(f"Published to {channel}: {message.event_type.value}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise
    
    async def start_listening(self):
        """Start listening for messages on subscribed channels"""
        if not self.subscribers:
            logger.warning("No subscribers registered")
            return
            
        # Subscribe to all channels
        for channel in self.subscribers.keys():
            await self.pubsub.subscribe(channel)
        
        self.running = True
        logger.info(f"Started listening on channels: {list(self.subscribers.keys())}")
        
        async for message in self.pubsub.listen():
            if not self.running:
                break
                
            if message["type"] == "message":
                try:
                    channel = message["channel"]
                    agent_message = AgentMessage.from_json(message["data"])
                    
                    # Call all callbacks for this channel
                    for callback in self.subscribers.get(channel, []):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(agent_message)
                            else:
                                callback(agent_message)
                        except Exception as e:
                            logger.error(f"Callback error on {channel}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
    
    # Context management methods
    async def store_context(self, session_id: str, context: InterviewContext):
        """Store interview context in Redis"""
        try:
            await self.redis_client.hset(
                "contexts",
                session_id,
                json.dumps(context.to_dict())
            )
            # Set expiration for 24 hours
            await self.redis_client.expire(f"contexts", 86400)
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            raise
    
    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve interview context from Redis"""
        try:
            context_json = await self.redis_client.hget("contexts", session_id)
            if context_json:
                return json.loads(context_json)
            return None
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None
    
    async def update_context_field(self, session_id: str, field_updates: Dict[str, Any]):
        """Update specific fields in stored context"""
        try:
            current_context = await self.get_context(session_id)
            if current_context:
                current_context.update(field_updates)
                await self.redis_client.hset(
                    "contexts",
                    session_id,
                    json.dumps(current_context)
                )
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
    
    # Agent discovery and health checking
    async def register_agent(self, agent_name: str, agent_info: Dict[str, Any]):
        """Register agent in discovery service"""
        agent_data = {
            "name": agent_name,
            "status": "active",
            "last_heartbeat": asyncio.get_event_loop().time(),
            **agent_info
        }
        await self.redis_client.hset("agents", agent_name, json.dumps(agent_data))
        await self.redis_client.expire("agents", 300)  # 5 minute TTL
    
    async def heartbeat(self, agent_name: str):
        """Send heartbeat for agent"""
        try:
            agent_json = await self.redis_client.hget("agents", agent_name)
            if agent_json:
                agent_data = json.loads(agent_json)
                agent_data["last_heartbeat"] = asyncio.get_event_loop().time()
                await self.redis_client.hset("agents", agent_name, json.dumps(agent_data))
        except Exception as e:
            logger.error(f"Heartbeat failed for {agent_name}: {e}")
    
    async def get_active_agents(self) -> List[str]:
        """Get list of currently active agents"""
        try:
            agents_data = await self.redis_client.hgetall("agents")
            current_time = asyncio.get_event_loop().time()
            active_agents = []
            
            for agent_name, agent_json in agents_data.items():
                agent_data = json.loads(agent_json)
                # Consider agent active if heartbeat within last 60 seconds
                if current_time - agent_data["last_heartbeat"] < 60:
                    active_agents.append(agent_name)
            
            return active_agents
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []
    
    # Message history and debugging
    async def get_message_history(self, channel: str, limit: int = 10) -> List[AgentMessage]:
        """Get recent message history for debugging"""
        try:
            messages_json = await self.redis_client.lrange(f"history:{channel}", 0, limit - 1)
            return [AgentMessage.from_json(msg) for msg in messages_json]
        except Exception as e:
            logger.error(f"Failed to get message history: {e}")
            return []

# Convenience channels for different types of communication
class Channels:
    COORDINATION = "coordination"  # Coordinator <-> All agents
    USER_INTERACTION = "user_interaction"  # User input/output
    CODE_ANALYSIS = "code_analysis"  # Code-related events
    EXECUTION = "execution"  # Code execution events
    EVALUATION = "evaluation"  # Solution assessment
    SYSTEM = "system"  # System-level events (startup, shutdown, errors)

# Message bus singleton for easy access
_message_bus_instance = None

async def get_message_bus(redis_url: str = None) -> MessageBus:
    """Get or create message bus singleton"""
    global _message_bus_instance
    if _message_bus_instance is None:
        redis_url = redis_url or "redis://localhost:6379"
        _message_bus_instance = MessageBus(redis_url)
        await _message_bus_instance.initialize()
    return _message_bus_instance