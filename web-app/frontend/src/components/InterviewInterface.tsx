import React, { useState, useEffect, useRef, useCallback } from 'react';
import { InterviewSession, WebSocketMessage } from '../types/index';
import { interviewAPI } from '../services/api';
import WebSocketService from '../services/websocket';
import CodeEditor from './CodeEditor';

interface InterviewInterfaceProps {
  session: InterviewSession;
  onSessionUpdate: (session: InterviewSession) => void;
  onError: (error: string) => void;
  onEnd: () => void;
}

const InterviewInterface: React.FC<InterviewInterfaceProps> = ({
  session,
  onSessionUpdate,
  onError,
  onEnd
}) => {
  const [messages, setMessages] = useState<any[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [code, setCode] = useState(session.current_problem?.starter_code || '');
  const [language, setLanguage] = useState('python');
  const [isLoading, setIsLoading] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const wsRef = useRef<WebSocketService | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'chat_message') {
      setMessages(prev => [...prev, message.data]);
    } else if (message.type === 'status_update') {
      // Handle status updates
      if (message.data.status) {
        onSessionUpdate({ ...session, status: message.data.status });
      }
    }
  }, [session, onSessionUpdate]);

  useEffect(() => {
    let isMounted = true;
    
    // Initialize WebSocket connection
    const ws = new WebSocketService(session.session_id);
    wsRef.current = ws;

    ws.addListener(handleWebSocketMessage);
    
    // Add slight delay to avoid React StrictMode issues
    const connectTimer = setTimeout(() => {
      if (isMounted) {
        ws.connect().then(() => {
          if (isMounted) {
            console.log('WebSocket connected for session:', session.session_id);
          }
        }).catch((error) => {
          if (isMounted) {
            console.error('WebSocket connection failed:', error);
            onError('Failed to connect to real-time updates');
          }
        });
      }
    }, 100);

    // Add initial problem message
    if (session.current_problem) {
      setMessages([{
        id: 'problem',
        speaker: 'interviewer',
        content: `Here's your problem: **${session.current_problem.title}**\n\n${session.current_problem.description}\n\nDifficulty: ${session.current_problem.difficulty}`,
        type: 'problem_intro',
        timestamp: new Date().toISOString()
      }]);
    }

    return () => {
      isMounted = false;
      clearTimeout(connectTimer);
      ws.disconnect();
    };
  }, [session.session_id, handleWebSocketMessage, onError, session.current_problem]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add keyboard shortcuts for code editor
  useEffect(() => {
    const handleRunCodeEvent = () => {
      handleSubmitCode();
    };

    const handleExecuteCodeEvent = () => {
      handleExecuteCode();
    };

    window.addEventListener('run-code', handleRunCodeEvent);
    window.addEventListener('execute-code', handleExecuteCodeEvent);

    return () => {
      window.removeEventListener('run-code', handleRunCodeEvent);
      window.removeEventListener('execute-code', handleExecuteCodeEvent);
    };
  }, []);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      speaker: 'user',
      content: currentMessage,
      type: 'message',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      await interviewAPI.sendMessage(session.session_id, {
        content: currentMessage
      });
    } catch (error: any) {
      onError(error.response?.data?.error || 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitCode = async () => {
    if (!code.trim() || isLoading) return;

    setIsLoading(true);
    try {
      await interviewAPI.submitCode(session.session_id, {
        code,
        language: language
      });
      
      const codeMessage = {
        id: Date.now().toString(),
        speaker: 'user',
        content: `\`\`\`${language}\n${code}\n\`\`\``,
        type: 'code_submission',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, codeMessage]);
    } catch (error: any) {
      onError(error.response?.data?.error || 'Failed to submit code');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecuteCode = async () => {
    if (isExecuting) return;

    setIsExecuting(true);
    try {
      const result = await interviewAPI.executeCode(session.session_id);
      
      const executionMessage = {
        id: Date.now().toString(),
        speaker: 'execution',
        content: `**Execution Result:**\n\`\`\`\n${result.output || 'No output'}\n\`\`\`\n${result.error ? `\n**Error:**\n\`\`\`\n${result.error}\n\`\`\`` : ''}`,
        type: 'execution_result',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, executionMessage]);
    } catch (error: any) {
      onError(error.response?.data?.error || 'Failed to execute code');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleEndInterview = async () => {
    if (window.confirm('Are you sure you want to end the interview?')) {
      try {
        await interviewAPI.endInterview(session.session_id);
        onEnd();
      } catch (error: any) {
        onError(error.response?.data?.error || 'Failed to end interview');
      }
    }
  };

  const renderMessage = (message: any) => {
    const isUser = message.speaker === 'user';
    const speakerIcons: Record<string, string> = {
      user: '👤',
      interviewer: '🤖',
      code_analyzer: '🔍',
      execution: '⚡'
    };
    const speakerIcon = speakerIcons[message.speaker] || '🤖';

    return (
      <div key={message.id} className={`message ${isUser ? 'user-message' : 'agent-message'}`}>
        <div className="message-header">
          <span className="speaker">
            {speakerIcon} {message.speaker}
          </span>
          <span className="timestamp">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div className="message-content">
          {message.content.split('\n').map((line: string, index: number) => (
            <div key={index}>
              {line.includes('```') ? (
                <pre><code>{line.replace(/```\w*/, '').replace('```', '')}</code></pre>
              ) : line.startsWith('**') && line.endsWith('**') ? (
                <strong>{line.slice(2, -2)}</strong>
              ) : (
                line
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="interview-interface">
      <div className="interview-header">
        <div className="session-info">
          <h2>Interview Session: {session.session_id.slice(0, 8)}</h2>
          <div className="status-indicators">
            <span className={`status-badge ${session.status}`}>
              {session.status}
            </span>
            <span className="difficulty-badge">
              {session.current_problem?.difficulty}
            </span>
          </div>
        </div>
        <button 
          onClick={handleEndInterview}
          className="end-button"
        >
          🔚 End Interview
        </button>
      </div>

      <div className="interview-content">
        <div className="chat-panel">
          <div className="messages-container">
            {messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </div>
          
          <form onSubmit={handleSendMessage} className="message-form">
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              placeholder="Ask questions, explain your approach..."
              disabled={isLoading}
              className="message-input"
            />
            <button 
              type="submit" 
              disabled={isLoading || !currentMessage.trim()}
              className="send-button"
            >
              {isLoading ? '⏳' : '📤'}
            </button>
          </form>
        </div>

        <div className="code-panel">
          <div className="code-header">
            <div className="code-header-left">
              <h3>💻 Code Editor</h3>
              <select 
                value={language} 
                onChange={(e) => setLanguage(e.target.value)}
                className="language-selector"
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="typescript">TypeScript</option>
                <option value="java">Java</option>
                <option value="cpp">C++</option>
                <option value="c">C</option>
                <option value="go">Go</option>
                <option value="rust">Rust</option>
              </select>
            </div>
            <div className="code-actions">
              <button 
                onClick={handleSubmitCode}
                disabled={isLoading || !code.trim()}
                className="submit-button"
              >
                {isLoading ? '⏳ Analyzing...' : '🔍 Submit for Review'}
              </button>
              <button 
                onClick={handleExecuteCode}
                disabled={isExecuting || !code.trim()}
                className="execute-button"
              >
                {isExecuting ? '⏳ Running...' : '▶️ Execute'}
              </button>
            </div>
          </div>
          
          <CodeEditor
            value={code}
            onChange={setCode}
            language={language}
            theme="vs-dark"
            height="400px"
            options={{
              placeholder: "Write your solution here...",
              scrollBeyondLastLine: false,
              minimap: { enabled: false },
              fontSize: 14,
              lineHeight: 20,
              wordWrap: 'on',
              contextmenu: true,
              quickSuggestions: true,
            }}
          />
          
          {session.current_problem?.hints && session.current_problem.hints.length > 0 && (
            <div className="hints-section">
              <h4>💡 Hints:</h4>
              <ul>
                {session.current_problem.hints.map((hint: string, index: number) => (
                  <li key={index}>{hint}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InterviewInterface;