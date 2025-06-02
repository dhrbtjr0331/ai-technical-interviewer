echo "🧹 Cleaning up AI Interview System..."

echo "Stopping all containers..."
docker-compose down

echo "Removing volumes (this will delete all data)..."
read -p "Are you sure you want to remove all data? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose down -v
    docker volume prune -f
    echo "✅ All data cleaned up"
else
    echo "ℹ️  Data volumes preserved"
fi

echo "Removing unused Docker images..."
docker image prune -f

echo "🎯 Cleanup complete!"