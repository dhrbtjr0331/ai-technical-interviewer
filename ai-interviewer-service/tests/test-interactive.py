import asyncio
import json
import redis.asyncio as redis
from datetime import datetime
import uuid

async def interactive_test():
    """Interactive test with hello world agent"""
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe("user_interaction")
    
    print("🎮 Interactive Hello World Agent Test")
    print("Type messages to send to the agent. Type 'quit' to exit.")
    
    async def listen_for_responses():
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    if data.get("source_agent") == "hello_world" and data.get("target_agent") == "user":
                        content = data["payload"]["content"]
                        print(f"\n🤖 Agent: {content}\n> ", end="", flush=True)
                except:
                    pass
    
    # Start listener task
    listener_task = asyncio.create_task(listen_for_responses())
    
    try:
        while True:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input:
                message = {
                    "event_type": "user_input",
                    "source_agent": "interactive_test",
                    "target_agent": "hello_world",
                    "payload": {
                        "content": user_input,
                        "session_id": "interactive_test_session"
                    },
                    "context_snapshot": {},
                    "timestamp": datetime.now().isoformat(),
                    "correlation_id": str(uuid.uuid4())
                }
                
                await r.publish("user_interaction", json.dumps(message))
                
    except KeyboardInterrupt:
        pass
    finally:
        listener_task.cancel()
        await r.close()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(interactive_test())