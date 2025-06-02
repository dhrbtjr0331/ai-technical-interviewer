-- Initialize PostgreSQL database for interview system

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sessions table - track interview sessions
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

-- Problems table - store curated coding problems
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

-- Session events table - audit trail of all interview events
CREATE TABLE session_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_agent VARCHAR(100)
);

-- Code submissions table - track all code changes
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

-- User performance table - track user progress over time
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

-- Indexes for performance
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
    'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.',
    'easy',
    '[
        {"input_data": {"nums": [2,7,11,15], "target": 9}, "expected_output": [0,1], "description": "nums[0] + nums[1] = 2 + 7 = 9"},
        {"input_data": {"nums": [3,2,4], "target": 6}, "expected_output": [1,2], "description": "nums[1] + nums[2] = 2 + 4 = 6"},
        {"input_data": {"nums": [3,3], "target": 6}, "expected_output": [0,1], "description": "nums[0] + nums[1] = 3 + 3 = 6"}
    ]',
    'def two_sum(nums, target):\n    # Your solution here\n    pass',
    '["Think about using a hash map to store values you''ve seen", "You can solve this in O(n) time", "For each number, check if target - number exists in your hash map"]',
    '["array", "hash-table"]'
),
(
    'valid-parentheses',
    'Valid Parentheses',
    'Given a string s containing just the characters ''('', '')'', ''{'', ''}'', ''['' and '']'', determine if the input string is valid. An input string is valid if: Open brackets must be closed by the same type of brackets. Open brackets must be closed in the correct order.',
    'easy',
    '[
        {"input_data": {"s": "()"}, "expected_output": true, "description": "Valid parentheses"},
        {"input_data": {"s": "()[]{}"}, "expected_output": true, "description": "Multiple valid pairs"},
        {"input_data": {"s": "(]"}, "expected_output": false, "description": "Mismatched brackets"},
        {"input_data": {"s": "([)]"}, "expected_output": false, "description": "Wrong order"}
    ]',
    'def is_valid(s):\n    # Your solution here\n    pass',
    '["Stack data structure is perfect for this", "Push opening brackets, pop when you see closing ones", "Make sure the popped bracket matches the closing bracket"]',
    '["string", "stack"]'
),
(
    'merge-two-sorted-lists',
    'Merge Two Sorted Lists',
    'You are given the heads of two sorted linked lists list1 and list2. Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists. Return the head of the merged linked list.',
    'easy',
    '[
        {"input_data": {"list1": [1,2,4], "list2": [1,3,4]}, "expected_output": [1,1,2,3,4,4], "description": "Merge two sorted lists"},
        {"input_data": {"list1": [], "list2": []}, "expected_output": [], "description": "Both empty lists"},
        {"input_data": {"list1": [], "list2": [0]}, "expected_output": [0], "description": "One empty list"}
    ]',
    'class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef merge_two_lists(list1, list2):\n    # Your solution here\n    pass',
    '["Use two pointers to traverse both lists", "Create a dummy node to simplify the logic", "Compare values and attach the smaller node"]',
    '["linked-list", "recursion"]'
),
(
    'best-time-to-buy-sell-stock',
    'Best Time to Buy and Sell Stock',
    'You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.',
    'easy',
    '[
        {"input_data": {"prices": [7,1,5,3,6,4]}, "expected_output": 5, "description": "Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5"},
        {"input_data": {"prices": [7,6,4,3,1]}, "expected_output": 0, "description": "No profit possible"},
        {"input_data": {"prices": [1,2]}, "expected_output": 1, "description": "Buy at 1, sell at 2"}
    ]',
    'def max_profit(prices):\n    # Your solution here\n    pass',
    '["Keep track of the minimum price seen so far", "Calculate profit for each day as current_price - min_price", "Keep track of the maximum profit seen so far"]',
    '["array", "dynamic-programming"]'
),
(
    'maximum-subarray',
    'Maximum Subarray',
    'Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.',
    'medium',
    '[
        {"input_data": {"nums": [-2,1,-3,4,-1,2,1,-5,4]}, "expected_output": 6, "description": "Subarray [4,-1,2,1] has the largest sum 6"},
        {"input_data": {"nums": [1]}, "expected_output": 1, "description": "Single element"},
        {"input_data": {"nums": [5,4,-1,7,8]}, "expected_output": 23, "description": "All elements sum"}
    ]',
    'def max_sub_array(nums):\n    # Your solution here\n    pass',
    '["This is the classic Kadane''s algorithm problem", "Keep track of current sum and maximum sum", "If current sum becomes negative, reset it to 0"]',
    '["array", "dynamic-programming", "divide-and-conquer"]'
);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ language 'plpgsql';

CREATE TRIGGER update_interview_sessions_updated_at BEFORE UPDATE ON interview_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_problems_updated_at BEFORE UPDATE ON problems
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();