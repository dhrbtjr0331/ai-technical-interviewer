#!/bin/bash
# complete-fix.sh - Fix all remaining issues

echo "🔧 Applying complete fixes..."

# 1. Fix docker-compose.yml version warning
echo "📝 Removing obsolete version from docker-compose.yml..."
if grep -q "^version:" docker-compose.yml; then
    sed -i.backup '/^version:/d' docker-compose.yml
    echo "✅ Removed version directive"
fi

# 2. Fix the SQL trigger syntax
echo "🗄️  Fixing SQL trigger syntax..."
cat > shared/database/init.sql << 'EOF'
-- shared/database/init.sql - FIXED VERSION
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sessions table
CREATE TABLE interview_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    current_problem_id VARCHAR(255),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    performance_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Problems table
CREATE TABLE problems (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    difficulty VARCHAR(20) NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    test_cases JSONB NOT NULL,
    starter_code TEXT DEFAULT '',
    hints JSONB DEFAULT '[]',
    topics JSONB DEFAULT '[]',
    time_limit_seconds INTEGER DEFAULT 3600,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session events table
CREATE TABLE session_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_agent VARCHAR(100)
);

-- Code submissions table
CREATE TABLE code_submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    code_content TEXT NOT NULL,
    is_valid_syntax BOOLEAN DEFAULT true,
    line_count INTEGER,
    approach_detected VARCHAR(255),
    execution_result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User performance table
CREATE TABLE user_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    problem_id VARCHAR(255) NOT NULL,
    solved BOOLEAN DEFAULT false,
    time_taken_seconds INTEGER,
    attempts INTEGER DEFAULT 1,
    hints_used INTEGER DEFAULT 0,
    difficulty_rating VARCHAR(20),
    performance_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, problem_id)
);

-- Indexes
CREATE INDEX idx_sessions_user_id ON interview_sessions(user_id);
CREATE INDEX idx_sessions_status ON interview_sessions(status);
CREATE INDEX idx_events_session_id ON session_events(session_id);
CREATE INDEX idx_events_type ON session_events(event_type);
CREATE INDEX idx_submissions_session_id ON code_submissions(session_id);
CREATE INDEX idx_performance_user_id ON user_performance(user_id);
CREATE INDEX idx_problems_difficulty ON problems(difficulty);

-- Insert sample problems
INSERT INTO problems (id, title, description, difficulty, test_cases, starter_code, hints, topics) VALUES 
(
    'two-sum',
    'Two Sum',
    'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.',
    'easy',
    '[
        {"input_data": {"nums": [2,7,11,15], "target": 9}, "expected_output": [0,1], "description": "nums[0] + nums[1] = 2 + 7 = 9"},
        {"input_data": {"nums": [3,2,4], "target": 6}, "expected_output": [1,2], "description": "nums[1] + nums[2] = 2 + 4 = 6"}
    ]',
    'def two_sum(nums, target):
    # Your solution here
    pass',
    '["Think about using a hash map", "You can solve this in O(n) time"]',
    '["array", "hash-table"]'
),
(
    'valid-parentheses', 
    'Valid Parentheses',
    'Given a string s containing just the characters (){}[] determine if the input string is valid.',
    'easy',
    '[
        {"input_data": {"s": "()"}, "expected_output": true, "description": "Valid parentheses"},
        {"input_data": {"s": "()[]{}"}, "expected_output": true, "description": "Multiple valid pairs"}
    ]',
    'def is_valid(s):
    # Your solution here
    pass',
    '["Stack data structure is perfect for this", "Push opening brackets, pop when you see closing ones"]',
    '["string", "stack"]'
);

-- Update timestamp trigger - FIXED SYNTAX  
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_interview_sessions_updated_at BEFORE UPDATE ON interview_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_problems_updated_at BEFORE UPDATE ON problems  
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
EOF

echo "✅ Fixed SQL syntax"

# 3. Clean restart with fixed files
echo "🧹 Clean restart..."
docker-compose down -v

# 4. Start PostgreSQL with fixed SQL
echo "🗄️  Starting PostgreSQL with fixed initialization..."
docker-compose up -d postgres

# 5. Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker-compose exec postgres pg_isready -U interview_user > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

# 6. Verify database was initialized correctly
echo "🧪 Verifying database initialization..."
PROBLEM_COUNT=$(docker-compose exec postgres psql -U interview_user -d interview_db -t -c "SELECT COUNT(*) FROM problems;" 2>/dev/null)

if [ $? -eq 0 ] && [ "${PROBLEM_COUNT// /}" -gt 0 ]; then
    echo "✅ Database initialized successfully! Found ${PROBLEM_COUNT// /} problems."
else
    echo "⚠️  Database initialization may have failed. Checking logs..."
    docker-compose logs postgres | tail -10
fi

# 7. Start Redis
echo "🔄 Starting Redis..."
docker-compose up -d redis
sleep 2

# 8. Test Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is working!"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# 9. Test container builds
echo "🏗️  Testing container builds..."
if docker-compose build coordinator > /dev/null 2>&1; then
    echo "✅ Coordinator builds successfully!"
else
    echo "❌ Coordinator build failed:"
    docker-compose build coordinator
fi

if docker-compose build cli > /dev/null 2>&1; then
    echo "✅ CLI builds successfully!"
else
    echo "❌ CLI build failed"
fi

echo ""
echo "🎉 System is ready!"
echo ""
echo "📋 Next steps:"
echo "1. Make sure your .env has CLAUDE_API_KEY set:"
echo "   echo \$CLAUDE_API_KEY"
echo ""
echo "2. Start the coordinator:"
echo "   docker-compose up -d coordinator"
echo ""
echo "3. Test the system:"
echo "   docker-compose run --rm cli python cli/main.py interactive"
echo ""
echo "📊 Current status:"
docker-compose ps