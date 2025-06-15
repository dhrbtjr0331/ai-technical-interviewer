import asyncio
import json
import redis.asyncio as redis
from datetime import datetime
import uuid

async def test_coordinator():
    """Test coordinator agent functionality"""
    print("🧪 Testing Coordinator Agent with CrewAI...")
    
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    pubsub = r.pubsub()
    
    # Subscribe to channels we want to monitor
    await pubsub.subscribe("coordination", "user_interaction")
    
    print("📡 Connected to Redis, starting tests...")
    
    # Test 1: Start interview
    print("\n1️⃣ Testing interview start...")
    start_message = {
        "event_type": "interview_state_change",
        "source_agent": "test_client",
        "target_agent": "coordinator",
        "payload": {
            "action": "start_interview",
            "session_id": "test_session_" + str(uuid.uuid4())[:8],
            "user_id": "test_user",
            "difficulty": "medium"
        },
        "context_snapshot": {},
        "timestamp": datetime.now().isoformat(),
        "correlation_id": str(uuid.uuid4())
    }