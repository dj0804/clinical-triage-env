# ============================================================
# Clinical Triage AI — OpenEnv Docker Image
# Base: python:3.11-slim
# Port: 7860 (HuggingFace Spaces convention)
# Build: docker build -t clinical-triage .
# Run:   docker run -p 7860:7860 clinical-triage
# ============================================================

FROM python:3.11-slim

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure Python can find local modules
ENV PYTHONPATH="/app"

# Switch to non-root user
USER appuser

# Expose HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
