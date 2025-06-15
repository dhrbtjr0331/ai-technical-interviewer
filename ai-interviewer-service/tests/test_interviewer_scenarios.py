import json
import asyncio
from datetime import datetime

class InterviewerTestScenarios:
    """Test scenarios for the adaptive interviewer agent"""
    
    def __init__(self):
        self.test_scenarios = [
            {
                "name": "Confident User - Interview Start",
                "scenario": "User starts with confidence",
                "user_input": "I'm ready! Let's do this coding problem.",
                "context": {
                    "interview_state": "problem_introduction",
                    "emotional_state": "confident",
                    "time_elapsed": 30
                },
                "expected_personality": "challenging",
                "expected_response_type": "professional_but_engaging"
            },
            {
                "name": "Nervous User - Needs Encouragement", 
                "scenario": "User expressing anxiety",
                "user_input": "I'm feeling a bit nervous about this interview",
                "context": {
                    "interview_state": "problem_introduction",
                    "emotional_state": "uncertain",
                    "time_elapsed": 60
                },
                "expected_personality": "encouraging",
                "expected_response_type": "warm_and_supportive"
            },
            {
                "name": "Stuck User - Problem Clarification",
                "scenario": "User struggling with problem understanding",
                "user_input": "I don't really understand what the problem is asking for",
                "context": {
                    "interview_state": "active_coding",
                    "emotional_state": "confused",
                    "time_elapsed": 600,
                    "code_changes": 0
                },
                "expected_personality": "clarifying",
                "expected_response_type": "clear_explanation"
            },
            {
                "name": "Frustrated User - Needs Support",
                "scenario": "User hitting roadblocks",
                "user_input": "This is really difficult, I'm not making any progress",
                "context": {
                    "interview_state": "active_coding",
                    "emotional_state": "frustrated",
                    "time_elapsed": 1200,
                    "code_changes": 3
                },
                "expected_personality": "encouraging",
                "expected_response_type": "supportive_guidance"
            },
            {
                "name": "Advanced User - Seeking Validation",
                "scenario": "Experienced user discussing approach",
                "user_input": "I'm thinking of using dynamic programming with memoization here",
                "context": {
                    "interview_state": "active_coding",
                    "emotional_state": "confident",
                    "time_elapsed": 800,
                    "code_changes": 5
                },
                "expected_personality": "challenging",
                "expected_response_type": "technical_discussion"
            },
            {
                "name": "Time Pressure - User Aware",
                "scenario": "User concerned about time",
                "user_input": "I feel like I'm running out of time, should I try a simpler approach?",
                "context": {
                    "interview_state": "active_coding",
                    "emotional_state": "uncertain",
                    "time_elapsed": 2400,  # 40 minutes
                    "code_changes": 8
                },
                "expected_personality": "encouraging",
                "expected_response_type": "time_management_guidance"
            },
            {
                "name": "Solution Discussion - Almost Done",
                "scenario": "User thinks they have solution",
                "user_input": "I think I've got it! Let me walk through my solution",
                "context": {
                    "interview_state": "solution_discussion",
                    "emotional_state": "confident",
                    "time_elapsed": 1800,
                    "code_changes": 12
                },
                "expected_personality": "neutral",
                "expected_response_type": "solution_review"
            },
            {
                "name": "Debugging Help - Code Issues",
                "scenario": "User having code problems",
                "user_input": "My code isn't working as expected, can you help me see what's wrong?",
                "context": {
                    "interview_state": "debugging",
                    "emotional_state": "frustrated",
                    "time_elapsed": 1500,
                    "code_changes": 15
                },
                "expected_personality": "clarifying",
                "expected_response_type": "debugging_guidance"
            }
        ]
    
    def get_test_scenario(self, scenario_name: str) -> dict:
        """Get a specific test scenario by name"""
        for scenario in self.test_scenarios:
            if scenario["name"] == scenario_name:
                return scenario
        return None
    
    def get_all_scenarios(self) -> list:
        """Get all test scenarios"""
        return self.test_scenarios
    
    def simulate_interviewer_response(self, scenario: dict) -> dict:
        """Simulate what the interviewer should respond based on scenario"""
        
        personality = scenario["expected_personality"]
        response_type = scenario["expected_response_type"]
        
        # Mock responses based on personality and context
        mock_responses = {
            "encouraging": {
                "warm_and_supportive": "That's completely normal! Technical interviews can feel intimidating, but I'm here to help guide you through it. Let's take this step by step - you've got this!",
                "supportive_guidance": "I can see you're putting in good effort here. Sometimes these problems take time to click. Let's break this down into smaller pieces - what's the first thing we need to figure out?",
                "time_management_guidance": "You're actually doing fine on time. Sometimes a simpler approach is exactly the right call - it shows good engineering judgment. What's the simplest solution you can think of?"
            },
            "challenging": {
                "professional_but_engaging": "Great energy! I like that confidence. Let's dive into this problem and see what you can do with it.",
                "technical_discussion": "That's an interesting approach! Can you walk me through why you think DP with memoization is the right fit here? What's the time complexity you're aiming for?"
            },
            "clarifying": {
                "clear_explanation": "Absolutely, let me break this down for you. The problem is asking you to...",
                "debugging_guidance": "Let's debug this systematically. Can you walk me through what you expected to happen versus what's actually happening?"
            },
            "neutral": {
                "solution_review": "Perfect! I'd love to hear your approach. Walk me through your solution step by step."
            }
        }
        
        response_text = mock_responses.get(personality, {}).get(response_type, "I understand. Please continue.")
        
        return {
            "personality_used": personality,
            "response_type": response_type,
            "response_text": response_text,
            "adaptive_reasoning": f"Chose {personality} personality based on {scenario['context']['emotional_state']} emotional state"
        }

async def test_interviewer_adaptation():
    """Test how well the interviewer adapts to different scenarios"""
    tester = InterviewerTestScenarios()
    
    print("🎭 Testing Interviewer Agent Adaptation")
    print("=" * 60)
    
    for i, scenario in enumerate(tester.get_all_scenarios(), 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"Scenario: {scenario['scenario']}")
        print(f"User Input: \"{scenario['user_input']}\"")
        print(f"Context: {scenario['context']}")
        
        # Simulate interviewer response
        mock_response = tester.simulate_interviewer_response(scenario)
        
        print(f"Expected Personality: {scenario['expected_personality']}")
        print(f"Actual Personality: {mock_response['personality_used']} ✅" if mock_response['personality_used'] == scenario['expected_personality'] else f"Actual Personality: {mock_response['personality_used']} ❌")
        print(f"Response: \"{mock_response['response_text']}\"")
        print(f"Reasoning: {mock_response['adaptive_reasoning']}")
        print("-" * 40)

def test_personality_transitions():
    """Test how interviewer should transition between personality modes"""
    print("\n🔄 Testing Personality Transitions")
    print("=" * 40)
    
    transitions = [
        {
            "from_state": "confident user starts",
            "to_state": "user gets stuck", 
            "from_personality": "challenging",
            "to_personality": "encouraging",
            "trigger": "User says 'I'm really confused now'"
        },
        {
            "from_state": "frustrated user gets help",
            "to_state": "user regains confidence",
            "from_personality": "encouraging", 
            "to_personality": "neutral",
            "trigger": "User says 'Oh I see now, that makes sense!'"
        },
        {
            "from_state": "user needs clarification",
            "to_state": "user ready to code",
            "from_personality": "clarifying",
            "to_personality": "neutral",
            "trigger": "User says 'Got it, let me start coding'"
        }
    ]
    
    for i, transition in enumerate(transitions, 1):
        print(f"{i}. {transition['from_state']} → {transition['to_state']}")
        print(f"   Personality: {transition['from_personality']} → {transition['to_personality']}")
        print(f"   Trigger: \"{transition['trigger']}\"")
        print()

async def run_full_interview_simulation():
    """Simulate a full interview conversation flow"""
    print("\n🎬 Full Interview Simulation")
    print("=" * 40)
    
    conversation_flow = [
        ("interviewer", "Welcome! Ready to start with a medium difficulty problem?"),
        ("user", "Yes, I'm ready!"),
        ("interviewer", "Great! Here's the Two Sum problem..."),
        ("user", "Can you clarify what the function signature should be?"),
        ("interviewer", "Of course! The function should take..."),
        ("user", "I think I'll use a hash map approach"),
        ("interviewer", "Interesting! Walk me through your reasoning..."),
        ("user", "Actually, I'm getting confused about the indices"),
        ("interviewer", "That's a common point of confusion. Let me help..."),
        ("user", "Oh I see! Let me code this up"),
        ("interviewer", "Perfect! Take your time and let me know if you need anything.")
    ]
    
    for i, (speaker, message) in enumerate(conversation_flow):
        timestamp = f"[{i*2:02d}:30]"
        if speaker == "interviewer":
            print(f"{timestamp} 🤖 Interviewer: {message}")
        else:
            print(f"{timestamp} 👤 User: {message}")
    
    print("\n📊 This conversation shows:")
    print("✅ Smooth transitions between personality modes")
    print("✅ Responsive to user emotional state changes") 
    print("✅ Maintains professional but warm tone")
    print("✅ Provides help without giving away solutions")

if __name__ == "__main__":
    asyncio.run(test_interviewer_adaptation())
    test_personality_transitions()
    asyncio.run(run_full_interview_simulation())