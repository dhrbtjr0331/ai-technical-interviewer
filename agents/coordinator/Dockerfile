FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ ./shared/
COPY problems/ ./problems/

# Copy agent-specific code
COPY agents/coordinator/ ./agents/coordinator/

# Set Python path
ENV PYTHONPATH=/app

# Run the coordinator agent
CMD ["python", "agents/coordinator/main.py"]