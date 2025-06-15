import React, { useState } from 'react';
import './App.css';
import InterviewInterface from './components/InterviewInterface';
import StartScreen from './components/StartScreen';
import { InterviewSession } from './types/index';

function App() {
  const [currentSession, setCurrentSession] = useState<InterviewSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 AI Technical Interviewer</h1>
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}
      </header>

      <main className="App-main">
        {currentSession ? (
          <InterviewInterface 
            session={currentSession}
            onSessionUpdate={setCurrentSession}
            onError={setError}
            onEnd={() => setCurrentSession(null)}
          />
        ) : (
          <StartScreen 
            onSessionStart={setCurrentSession}
            onError={setError}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        )}
      </main>

    </div>
  );
}

export default App;
