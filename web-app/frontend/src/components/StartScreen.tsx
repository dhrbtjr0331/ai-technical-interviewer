import React, { useState } from 'react';
import { interviewAPI } from '../services/api';
import { InterviewSession } from '../types/index';

interface StartScreenProps {
  onSessionStart: (session: InterviewSession) => void;
  onError: (error: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const StartScreen: React.FC<StartScreenProps> = ({
  onSessionStart,
  onError,
  isLoading,
  setIsLoading
}) => {
  const [userId, setUserId] = useState('');
  const [difficulty, setDifficulty] = useState('medium');

  const handleStartInterview = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!userId.trim()) {
      onError('Please enter your name');
      return;
    }

    setIsLoading(true);
    try {
      const response = await interviewAPI.startInterview({
        user_id: userId.trim(),
        difficulty,
      });

      // Get the full session details
      const session = await interviewAPI.getInterviewStatus(response.session_id);
      onSessionStart(session);
    } catch (error: any) {
      onError(error.response?.data?.error || 'Failed to start interview');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="start-screen">
      <div className="start-container">
        <div className="welcome-section">
          <h2>Welcome to the AI Technical Interview</h2>
          <p>
            This is an AI-powered technical interview system that simulates a real 
            coding interview experience. You'll work with multiple AI agents:
          </p>
          <ul>
            <li>🤖 <strong>Interviewer</strong> - Guides conversation and provides feedback</li>
            <li>🔍 <strong>Code Analyzer</strong> - Reviews your code in real-time</li>
            <li>⚡ <strong>Execution Engine</strong> - Runs and tests your solutions</li>
          </ul>
        </div>

        <form onSubmit={handleStartInterview} className="start-form">
          <div className="form-group">
            <label htmlFor="userId">Your Name:</label>
            <input
              id="userId"
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter your name"
              required
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="difficulty">Difficulty Level:</label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={isLoading}
            >
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <button 
            type="submit" 
            className="start-button"
            disabled={isLoading}
          >
            {isLoading ? '🔄 Starting Interview...' : '🚀 Start Interview'}
          </button>
        </form>

        <div className="tips-section">
          <h3>Tips for Success:</h3>
          <ul>
            <li>Think out loud - explain your approach</li>
            <li>Ask clarifying questions if needed</li>
            <li>Write clean, readable code</li>
            <li>Test your solution as you go</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default StartScreen;