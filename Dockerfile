FROM python:3.9-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--workers", "2", "--bind", "0.0.0.0:8000", "--timeout", "120"]