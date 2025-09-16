FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

# Use gunicorn with uvicorn workers for better production handling
RUN pip install gunicorn

CMD ["sh", "-c", "exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8080} --access-logfile - --error-logfile -"]
