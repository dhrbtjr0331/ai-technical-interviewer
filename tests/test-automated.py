import asyncio
import json
import redis.asyncio as redis
from datetime import datetime
import uuid

async def automated_test():
    """Automated test with hello world agent"""
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe("user_interaction")
    
    print("🤖 Starting automated Hello World Agent test...")
    
    # Send a test message
    test_message = {
        "event_type": "user_input",
        "source_agent": "automated_test",
        "target_agent": "hello_world",
        "payload": {
            "content": "Hello! This is an automated test. Can you respond?",
            "session_id": "automated_test_session"
        },
        "context_snapshot": {},
        "timestamp": datetime.now().isoformat(),
        "correlation_id": str(uuid.uuid4())
    }
    
    print(f"📤 Sending test message: {test_message['payload']['content']}")
    await r.publish("user_interaction", json.dumps(test_message))
    
    # Listen for response with timeout
    print("👂 Listening for response...")
    timeout = 15
    start_time = asyncio.get_event_loop().time()
    response_received = False
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])
                if data.get("source_agent") == "hello_world":
                    content = data["payload"]["content"]
                    print(f"✅ Got response from agent: {content}")
                    response_received = True
                    break
            except Exception as e:
                print(f"⚠️ Error parsing message: {e}")
                
        if asyncio.get_event_loop().time() - start_time > timeout:
            print("⏰ Timeout waiting for response")
            break
    
    await r.aclose()
    
    if response_received:
        print("🎉 Test PASSED: Agent responded successfully!")
        return True
    else:
        print("❌ Test FAILED: No response from agent")
        return False

if __name__ == "__main__":
    success = asyncio.run(automated_test())
    exit(0 if success else 1)