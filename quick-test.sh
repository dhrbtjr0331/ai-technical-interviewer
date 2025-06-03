echo "🧪 Running quick system test..."

# Test 1: Start interview
echo "1️⃣ Starting interview session..."
SESSION_OUTPUT=$(docker-compose run --rm cli python cli/main.py start --difficulty easy --user-id test_user)
echo "$SESSION_OUTPUT"

sleep 2

# Test 2: Send user input
echo "2️⃣ Sending user input..."
docker-compose run --rm cli python cli/main.py input "Hello, can you explain the first problem?"

sleep 2

# Test 3: Check status
echo "3️⃣ Checking system status..."
docker-compose run --rm cli python cli/main.py status

sleep 1

# Test 4: Submit simple code
echo "4️⃣ Submitting test code..."
docker-compose run --rm cli python cli/main.py code "def two_sum(nums, target): return [0, 1]"

sleep 2

# Test 5: Execute code
echo "5️⃣ Executing code..."
docker-compose run --rm cli python cli/main.py execute

sleep 3

# Test 6: End interview
echo "6️⃣ Ending interview..."
docker-compose run --rm cli python cli/main.py end

echo "✅ Quick test complete!"

# ===================================
# dev-logs.sh - Development logging script
#!/bin/bash

echo "📊 Starting log monitoring..."
echo "Press Ctrl+C to stop"

# Open logs in parallel
docker-compose logs -f coordinator &
COORD_PID=$!

docker-compose logs -f interviewer &
INTERVIEW_PID=$!

docker-compose logs -f code_analyzer &
CODE_PID=$!

docker-compose logs -f execution &
EXEC_PID=$!

# Wait for interrupt
trap 'kill $COORD_PID $INTERVIEW_PID $CODE_PID $EXEC_PID' INT
wait