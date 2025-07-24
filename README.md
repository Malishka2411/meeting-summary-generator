# 📝 Meeting Summary Generator

A simple web app that takes meeting transcripts (text/audio) and generates:
- ✅ Key point summaries
- ✅ Action items
- ✅ Sentiment analysis

Built using **Streamlit**, **Whisper**, and **Transformers**.

---

## 🔧 Features

- 📄 Accepts `.txt`, `.docx`, or audio files (`.mp3`, `.wav`)
- 🧠 Summarizes content using BART (`facebook/bart-large-cnn`)
- 📌 Extracts key action items from summary
- 📊 Analyzes sentiment using BERT
- 🔈 Audio support powered by OpenAI Whisper
- 💾 Download summary as `.txt`

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Malishka2411/meeting-summary-generator.git
cd meeting-summary-generator
````

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📁 File Structure

```bash
meeting-summary-generator/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .gitignore           # Git exclusions
└── README.md            # This file
```

---

## 🛠 Tech Stack

* **Python**
* **Streamlit**
* **Hugging Face Transformers**
* **Whisper (OpenAI)**
* **Torch**

---

## 🙋‍♀️ Author

Made with ❤️ by [Malishka Paul](https://github.com/Malishka2411)

---
