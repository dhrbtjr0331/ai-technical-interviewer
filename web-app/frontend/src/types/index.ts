export interface InterviewSession {
  session_id: string;
  user_id: string;
  status: string;
  current_problem?: Problem;
  performance_metrics: Record<string, any>;
  messages: ChatMessage[];
}

export interface Problem {
  id: string;
  title: string;
  description: string;
  difficulty: string;
  starter_code: string;
  hints: string[];
  topics: string[];
  test_cases: TestCase[];
}

export interface TestCase {
  input_data: any;
  expected_output: any;
  description: string;
  is_hidden: boolean;
}

export interface ChatMessage {
  id: string;
  speaker: string;
  content: string;
  timestamp: string;
  type: string;
}

export interface StartInterviewRequest {
  user_id: string;
  difficulty?: string;
}

export interface SendMessageRequest {
  content: string;
}

export interface SubmitCodeRequest {
  code: string;
  language?: string;
}

export interface ExecuteCodeResponse {
  output: string;
  success: boolean;
  error?: string;
  test_cases?: TestCaseResult[];
  metrics?: Record<string, any>;
}

export interface TestCaseResult {
  input: any;
  expected: any;
  actual: any;
  passed: boolean;
}

export interface WebSocketMessage {
  type: string;
  session_id?: string;
  data: any;
  timestamp: string;
}