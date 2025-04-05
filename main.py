import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

# Lightweight model for Render
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

app = FastAPI()

@app.post("/summarize")
async def summarize(req: Request):
    try:
        data = await req.json()
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        result = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return {"summary": result[0]["summary_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
