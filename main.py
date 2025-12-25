import os
import redis
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
from typing import Optional
from prometheus_fastapi_instrumentator import Instrumentator

REDIS_HOST = os.getenv("REDIS_HOST", "redis-db")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
HISTORY_LIMIT = 1000

app = FastAPI(title="NLP Microservice with Redis")

Instrumentator().instrument(app).expose(app)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

_models = {}

async def get_model(task_name: str, model_name: str):
    """
    Загружает модель только когда она нужна первый раз.
    """
    if task_name not in _models:
        print(f"⏳ Loading model for {task_name}...")
        loop = asyncio.get_running_loop()
        _models[task_name] = await loop.run_in_executor(
            None, 
            lambda: pipeline(task_name, model=model_name)
        )
        print(f"✅ Model {task_name} loaded!")
    return _models[task_name]

class APIInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, title="Input text", description="Text to analyze/translate")

def save_log(task, text, result):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "task": task,
            "input": text,
            "result": str(result)
        }
        r.lpush("api_logs", json.dumps(log_data))
        r.ltrim("api_logs", 0, HISTORY_LIMIT - 1)
    except Exception as e:
        print(f"Redis log error: {e}")

@app.get("/")
def home():
    return {"status": "Online", "db_status": "Connected to Redis"}

@app.get("/history")
def get_history():
    try:
        logs = r.lrange("api_logs", 0, 9)
        return [json.loads(log) for log in logs]
    except Exception as e:
        return {"error": str(e)}

@app.post("/sentiment")
async def predict_sentiment(data: APIInput):
    model = await get_model("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
    result = model(data.text)
    save_log("SENTIMENT", data.text, result)
    return {"result": result}

@app.post("/translate")
async def translate_text(data: APIInput):
    model = await get_model("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr")
    result = model(data.text)
    translated_text = result[0]['translation_text']
    save_log("TRANSLATION", data.text, translated_text)
    return {"translated_text": translated_text}