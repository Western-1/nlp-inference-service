import os
import redis
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="NLP Microservice with Redis")

r = redis.Redis(host='redis-db', port=6379, decode_responses=True)

print("Loading models...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

class APIInput(BaseModel):
    text: str

def save_log(task, text, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "timestamp": timestamp,
        "task": task,
        "input": text,
        "result": str(result)
    }
    r.lpush("api_logs", json.dumps(log_data))

@app.get("/")
def home():
    return {"status": "Online", "db_status": "Connected to Redis"}

@app.get("/history")
def get_history():
    logs = r.lrange("api_logs", 0, 9)
    return [json.loads(log) for log in logs]

@app.post("/sentiment")
def predict_sentiment(data: APIInput):
    result = sentiment_pipeline(data.text)
    save_log("SENTIMENT", data.text, result)
    return {"result": result}

@app.post("/translate")
def translate_text(data: APIInput):
    result = translator(data.text)
    translated_text = result[0]['translation_text']
    save_log("TRANSLATION", data.text, translated_text)
    return {"translated_text": translated_text}
