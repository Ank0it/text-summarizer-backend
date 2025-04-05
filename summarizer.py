# backend/summarizer.py

from transformers import pipeline

# Load model once at startup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    if len(text.strip()) == 0:
        return "No text provided for summarization."

    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']
