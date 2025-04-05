from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    text = body.get("text")

    if not text or len(text.strip()) < 20:
        return {"summary": "Please enter a longer text to summarize."}

    summary = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return {"summary": summary[0]['summary_text']}
