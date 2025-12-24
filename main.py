import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Setup logging directory
if not os.path.exists("logs"):
    os.makedirs("logs")

app = FastAPI(title="NLP Inference Service")

print("Loading models, please wait...")

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

translation_pipeline = pipeline(
    "translation_en_to_fr", 
    model="Helsinki-NLP/opus-mt-en-fr"
)

print("Models loaded successfully.")


class APIInput(BaseModel):
    text: str


def save_log(task_name: str, input_text: str, output_result: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] TASK: {task_name} | INPUT: {input_text} | OUTPUT: {output_result}\n"
    
    with open("logs/service_history.log", "a", encoding="utf-8") as f:
        f.write(log_entry)


@app.get("/")
def health_check():
    return {"status": "healthy", "available_services": ["/sentiment", "/translate"]}


@app.post("/sentiment")
def predict_sentiment(data: APIInput):
    result = sentiment_pipeline(data.text)
    save_log("SENTIMENT", data.text, str(result))
    return {"result": result}


@app.post("/translate")
def translate_text(data: APIInput):
    result = translation_pipeline(data.text)
    translated_text = result[0]['translation_text']
    
    save_log("TRANSLATION", data.text, translated_text)
    return {"translated_text": translated_text}
