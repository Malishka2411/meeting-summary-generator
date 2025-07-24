# meeting_summary_app/app.py

import streamlit as st
st.set_page_config(page_title="Meeting Summary Generator", layout="centered")

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from docx import Document
import tempfile
import os
import whisper

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    whisper_model = whisper.load_model("base")
    return summarizer, sentiment_model, sentiment_tokenizer, whisper_model

summarizer, sentiment_model, sentiment_tokenizer, whisper_model = load_models()

def get_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(scores).item()
    stars = predicted_class + 1
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positive"


def generate_summary(text):
    max_chunk = 1000
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summarized_chunks = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    summary = "\n".join([chunk['summary_text'] for chunk in summarized_chunks])
    return summary

def extract_action_items(summary):
    lines = summary.split(". ")
    return [f"- {line.strip()}" for line in lines if line.strip() and ("should" in line.lower() or "need to" in line.lower() or "must" in line.lower())]

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    result = whisper_model.transcribe(tmp_path)
    return result['text']

def download_summary(summary_block):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(summary_block.encode('utf-8'))
        tmp_path = tmp.name
    return tmp_path

# Streamlit UI
st.title("📝 Meeting Summary Generator")

uploaded_text = st.file_uploader("Upload a transcript (.txt or .docx)", type=["txt", "docx"])
uploaded_audio = st.file_uploader("Or upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

if uploaded_text or uploaded_audio:
    with st.spinner("Processing input..."):
        if uploaded_audio:
            raw_text = transcribe_audio(uploaded_audio)
        else:
            raw_text = read_file(uploaded_text)

        summary = generate_summary(raw_text)
        action_items = extract_action_items(summary)
        sentiment = get_sentiment(summary)

        st.subheader("📌 Summary")
        st.text_area("Key Points", summary, height=200)

        st.subheader("✅ Action Items")
        st.markdown("\n".join(action_items) if action_items else "No clear action items detected.")

        st.subheader("📊 Sentiment")
        st.markdown(f"**Overall Tone:** {sentiment}")

        # Prepare block for download
        summary_block = f"Meeting Summary\n\nKey Points:\n{summary}\n\nAction Items:\n" + "\n".join(action_items) + f"\n\nSentiment:\n{sentiment}"
        tmp_path = download_summary(summary_block)

        with open(tmp_path, "rb") as file:
            st.download_button(
                label="📥 Download Summary as .txt",
                data=file,
                file_name="meeting_summary.txt",
                mime="text/plain"
            )
