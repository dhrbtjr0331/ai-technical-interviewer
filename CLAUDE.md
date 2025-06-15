# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repository contains:
- **ai-interviewer-service/**: Multi-agent AI technical interview system (microservice)
- **web-app/**: Web application with Go BFF and React frontend

## Architecture Overview

This is an AI-powered technical interview system built with a multi-agent architecture using CrewAI, LangChain, and Docker. The system orchestrates real-time technical interviews with specialized agents handling different aspects of the interview process.

### Core Components

**Agent Architecture:**
- **Coordinator Agent**: Master orchestrator that manages interview flow, routes messages, and controls state transitions using CrewAI
- **Interviewer Agent**: Handles conversational aspects and adaptive interview flow
- **Code Analyzer Agent**: Provides real-time code analysis and feedback (temperature: 0.3)
- **Execution Agent**: Securely executes code using Docker-in-Docker (temperature: 0.2)
- **Hello World Agent**: Test agent for system validation

**Infrastructure:**
- **Message Bus**: Redis-based pub/sub system for agent communication with channels for coordination, user interaction, code analysis, execution, evaluation, and system events
- **Database**: PostgreSQL for session persistence and problem storage
- **Interview Context**: Comprehensive state management including code history, conversation history, performance metrics, and agent memory

**Interview Flow:**
- State machine with phases: initializing → problem_intro → clarification → active_coding → debugging → execution_review → solution_discussion → optimization → completed
- Real-time code analysis with syntax validation and approach detection
- Performance tracking (time, code changes, execution attempts, hints used, questions asked)

## Essential Commands

**AI Interviewer Service Management:**
```bash
cd ai-interviewer-service
make setup          # Initial system setup
make start           # Start all services  
make stop            # Stop all services
make restart         # Restart all services
make status          # Check service status
make build           # Rebuild all containers
```

**Development:**
```bash
make interactive     # Start interactive CLI
make test            # Run quick functionality test
make logs            # Monitor all logs
make clean           # Clean up all resources
```

**Service-Specific Logs:**
```bash
make coord-logs      # Coordinator agent logs
make interview-logs  # Interviewer agent logs  
make code-logs       # Code analyzer logs
make exec-logs       # Execution agent logs
```

**Direct Docker Commands:**
```bash
docker-compose up -d                    # Start services
docker-compose logs -f [service_name]   # Follow specific service logs
docker-compose run --rm cli python cli/main.py interactive  # Interactive mode
```

## Development Notes

**Environment Variables Required:**
- `CLAUDE_API_KEY`: Anthropic API key for LLM integration
- `LLM_MODEL`: Model selection (default: claude-3-5-sonnet-20240620)
- `LLM_TEMPERATURE`: Temperature settings vary by agent (coordinator: 0.7, code_analyzer: 0.3, execution: 0.2)

**Database Schema:**
- Problems table with fields: id, title, description, difficulty, starter_code, hints, topics
- Session persistence handled through Redis context storage

**Key Dependencies:**
- CrewAI 0.126.0 for agent orchestration
- LangChain 0.3.x ecosystem for LLM integration  
- Redis for message bus and context storage
- PostgreSQL for problem and session data
- Docker-in-Docker for secure code execution

**Testing:**
- Use `make test` for quick functionality validation
- Individual agent testing files available in root directory
- CLI interface available for interactive testing

## Problem Management

Problems are stored as JSON files in the `/problems` directory and loaded into PostgreSQL. Each problem includes:
- Test cases with input/output validation
- Difficulty levels (easy/medium/hard)
- Starter code templates
- Hint systems
- Topic categorization

## Agent Communication

Agents communicate through Redis pub/sub channels with structured message types (AgentMessage) containing event types, routing information, payloads, and context snapshots. The coordinator routes messages based on content analysis and interview state.