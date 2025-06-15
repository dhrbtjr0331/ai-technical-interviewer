import axios from 'axios';
import {
  InterviewSession,
  StartInterviewRequest,
  SendMessageRequest,
  SubmitCodeRequest,
  ExecuteCodeResponse,
  Problem
} from '../types/index';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8081/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const interviewAPI = {
  // Start new interview
  startInterview: async (request: StartInterviewRequest) => {
    const response = await api.post('/interviews', request);
    return response.data;
  },

  // Get interview status
  getInterviewStatus: async (sessionId: string): Promise<InterviewSession> => {
    const response = await api.get(`/interviews/${sessionId}`);
    return response.data;
  },

  // End interview
  endInterview: async (sessionId: string) => {
    const response = await api.delete(`/interviews/${sessionId}`);
    return response.data;
  },

  // Send message to interviewer
  sendMessage: async (sessionId: string, request: SendMessageRequest) => {
    const response = await api.post(`/interviews/${sessionId}/messages`, request);
    return response.data;
  },

  // Submit code for analysis
  submitCode: async (sessionId: string, request: SubmitCodeRequest) => {
    const response = await api.post(`/interviews/${sessionId}/code`, request);
    return response.data;
  },

  // Execute code
  executeCode: async (sessionId: string): Promise<ExecuteCodeResponse> => {
    const response = await api.post(`/interviews/${sessionId}/execute`);
    return response.data;
  },

  // Get available problems
  getProblems: async (): Promise<{ problems: Problem[] }> => {
    const response = await api.get('/problems');
    return response.data;
  },
};

export default api;