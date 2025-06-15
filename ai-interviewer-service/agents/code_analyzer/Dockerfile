FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY agents/code_analyzer/ ./agents/code_analyzer/

ENV PYTHONPATH=/app

CMD ["python", "agents/code_analyzer/main.py"]
