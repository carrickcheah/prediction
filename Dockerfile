FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy pyproject.toml first for better caching
COPY app/pyproject.toml .

# Initialize UV project and sync dependencies
RUN uv init --no-readme && \
    uv sync

# Copy application code
COPY app/src ./src
COPY app/scripts ./scripts

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "python", "src/main.py", "run"]