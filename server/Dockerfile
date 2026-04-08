# LegalMind-RL — server Dockerfile
# Build context must be the project root (parent of server/)
# docker build -f server/Dockerfile -t legalmind-rl .

FROM python:3.11-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
COPY models.py   ./models.py
COPY reward.py   ./reward.py
COPY tasks.py    ./tasks.py
COPY graders.py  ./graders.py
COPY openenv.yaml ./openenv.yaml
COPY server/     ./server/

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
