# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Cloud Run expected port
EXPOSE 8080

# Start FastAPI with uvicorn, binding to Cloud Run's $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
