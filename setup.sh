set -e

echo "🚀 Setting up AI Technical Interview System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Claude API key before continuing."
    exit 1
fi

# Source environment variables
source .env

# Validate Claude API key
if [ "$CLAUDE_API_KEY" = "your_claude_api_key_here" ]; then
    echo "❌ Please set your CLAUDE_API_KEY in .env file"
    exit 1
fi

echo "🔧 Building Docker containers..."
docker-compose build

echo "🗄️  Starting database and Redis..."
docker-compose up -d redis postgres

echo "⏳ Waiting for services to be ready..."
sleep 10

echo "✅ Running database initialization..."
docker-compose exec postgres psql -U interview_user -d interview_db -f /docker-entrypoint-initdb.d/init.sql

echo "🎯 Starting all agents..."
docker-compose up -d

echo "⏳ Waiting for agents to initialize..."
sleep 5

echo "🧪 Testing system health..."
docker-compose ps

echo ""
echo "✨ Setup complete! Your AI Interview system is running."
echo ""
echo "🎮 Quick start commands:"
echo "  # Interactive mode"
echo "  docker-compose run --rm cli python cli/main.py interactive"
echo ""
echo "  # Or individual commands:"
echo "  docker-compose run --rm cli python cli/main.py start --difficulty medium"
echo "  docker-compose run --rm cli python cli/main.py input 'Can you explain the problem?'"
echo "  docker-compose run --rm cli python cli/main.py code 'def solution(): pass'"
echo "  docker-compose run --rm cli python cli/main.py execute"
echo "  docker-compose run --rm cli python cli/main.py status"
echo ""
echo "📊 Monitor logs:"
echo "  docker-compose logs -f coordinator"
echo "  docker-compose logs -f interviewer"
echo ""
echo "🔧 Useful development commands:"
echo "  docker-compose down    # Stop all services"
echo "  docker-compose up -d   # Start all services"
echo "  docker-compose ps      # Check service status"