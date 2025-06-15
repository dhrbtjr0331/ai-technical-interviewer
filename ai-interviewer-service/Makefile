.PHONY: setup test start stop logs clean interactive help

help: ## Show this help message
	@echo "AI Technical Interview System - Make Commands"
	@echo "==============================================" 
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Initial system setup
	@chmod +x setup.sh && ./setup.sh

test: ## Run quick functionality test
	@chmod +x quick-test.sh && ./quick-test.sh

start: ## Start all services
	@docker-compose up -d
	@echo "✅ All services started"

stop: ## Stop all services
	@docker-compose down
	@echo "🛑 All services stopped"

logs: ## Monitor all logs
	@chmod +x dev-logs.sh && ./dev-logs.sh

clean: ## Clean up all resources
	@chmod +x cleanup.sh && ./cleanup.sh

interactive: ## Start interactive CLI
	@docker-compose run --rm cli python cli/main.py interactive

status: ## Check service status
	@docker-compose ps

build: ## Rebuild all containers
	@docker-compose build

restart: stop start ## Restart all services

# Development shortcuts
coord-logs: ## Show coordinator logs
	@docker-compose logs -f coordinator

interview-logs: ## Show interviewer logs  
	@docker-compose logs -f interviewer

code-logs: ## Show code analyzer logs
	@docker-compose logs -f code_analyzer

exec-logs: ## Show execution logs
	@docker-compose logs -f execution