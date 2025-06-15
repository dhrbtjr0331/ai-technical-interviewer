set -e

echo "🧪 Testing Hello World Agent..."

# Make sure infrastructure is running
echo "1️⃣ Starting infrastructure..."
docker-compose up -d redis postgres

echo "⏳ Waiting for services..."
sleep 5

# Build and start hello world agent
echo "2️⃣ Building and starting hello world agent..."
docker-compose up -d hello_world

echo "⏳ Waiting for agent initialization..."
sleep 10

# Test 1: Check agent is registered
echo "3️⃣ Testing agent registration..."
docker-compose exec redis redis-cli HGET agents hello_world

# Test 2: Send a test message
echo "4️⃣ Sending test message..."

# Create a simple test script to send message via Redis
cat > test_message.py << 'EOF'
import asyncio
import json
import redis.asyncio as redis
from datetime import datetime
import uuid

async def send_test_message():
    r = redis.from_url("redis://redis:6379", decode_responses=True)
    
    message = {
        "event_type": "user_input",
        "source_agent": "test_client",
        "target_agent": "hello_world",
        "payload": {
            "content": "Hello! Can you hear me?",
            "session_id": "test_session_123"
        },
        "context_snapshot": {},
        "timestamp": datetime.now().isoformat(),
        "correlation_id": str(uuid.uuid4())
    }
    
    await r.publish("user_interaction", json.dumps(message))
    print("✅ Test message sent!")
    
    # Listen for response
    pubsub = r.pubsub()
    await pubsub.subscribe("user_interaction")
    
    print("👂 Listening for response...")
    timeout = 10
    start_time = asyncio.get_event_loop().time()
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            if data.get("source_agent") == "hello_world":
                print(f"🎉 Got response: {data['payload']['content']}")
                break
                
        if asyncio.get_event_loop().time() - start_time > timeout:
            print("⏰ Timeout waiting for response")
            break
    
    await r.close()

if __name__ == "__main__":
    asyncio.run(send_test_message())
EOF

# Run the test using a temporary container on the same network
docker run --rm --network ai-technical-interviewer_default -v $(pwd)/test_message.py:/test_message.py python:3.11-slim bash -c "pip install redis && python /test_message.py"

# Test 3: Check logs
echo "5️⃣ Checking agent logs..."
docker-compose logs --tail=20 hello_world

# Test 4: Check active agents
echo "6️⃣ Checking active agents..."
docker-compose exec redis redis-cli HGETALL agents

# Cleanup test file
rm -f test_message.py

echo ""
echo "✅ Hello World Agent test complete!"
echo ""
echo "🎯 Next steps:"
echo "  - Check logs: docker-compose logs -f hello_world"
echo "  - Test with CLI: docker-compose run --rm cli python cli/main.py interactive"
echo "  - Monitor Redis: docker-compose exec redis redis-cli MONITOR"