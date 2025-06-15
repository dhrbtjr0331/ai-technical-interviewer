# AI Technical Interviewer Service

A multi-agent AI system that conducts technical interviews using CrewAI, LangChain, and Docker.

## Architecture

This service consists of specialized agents that work together to conduct technical interviews:

- **Coordinator Agent**: Orchestrates interview flow and routes messages
- **Interviewer Agent**: Handles conversational aspects and adaptive interview flow  
- **Code Analyzer Agent**: Provides real-time code analysis and feedback
- **Execution Agent**: Securely executes code using Docker-in-Docker
- **Hello World Agent**: Test agent for system validation

## Quick Start

```bash
# Setup and start all services
make setup
make start

# Interactive CLI mode
make interactive

# View logs
make logs

# Stop services
make stop
```

## Directory Structure

```
ai-interviewer-service/
├── agents/           # Agent implementations
├── cli/             # Command-line interface
├── problems/        # Interview problem definitions
├── scripts/         # Setup and utility scripts
├── shared/          # Shared models and message bus
├── tests/          # Test files
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Environment Variables

- `CLAUDE_API_KEY`: Anthropic API key for LLM integration
- `LLM_MODEL`: Model selection (default: claude-3-5-sonnet-20240620)
- `LLM_TEMPERATURE`: Temperature settings vary by agent

## API Integration

This service communicates via Redis pub/sub channels and can be integrated with external applications through the message bus interface.

See `shared/message_bus.py` and `shared/models.py` for integration details.