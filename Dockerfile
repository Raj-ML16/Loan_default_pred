# Simplified Dockerfile for ML Loan Prediction API
# Fixed to use standalone FastAPI app

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (using pre-compiled wheels to avoid gcc)
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=appuser:appuser . .

# Create directory for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE $APP_PORT

# Health check (simplified)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# FIXED: Use the standalone FastAPI file
CMD ["python", "fastapi_standalone.py"]