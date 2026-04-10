# Sentiment Analysis using BERT

## Overview
This project is an AI-based sentiment analyzer that classifies text as positive or negative using a pre-trained BERT model.

It also includes a simple web interface built using Streamlit for real-time predictions.

This project uses a transformer-based deep learning model (BERT) to achieve high accuracy in sentiment classification.

---

## Tech Stack
- Python
- Transformers (HuggingFace)
- PyTorch
- Pandas
- Scikit-learn
- Streamlit

---

## How It Works
1. Input text is given by the user
2. Text is tokenized using BERT tokenizer
3. Model predicts sentiment
4. Output is displayed

---

## Setup & Run

### Create virtual environment
python -m venv bertenv

### Activate environment
bertenv\Scripts\activate

### Install dependencies
pip install streamlit transformers torch pandas scikit-learn

### Run the app
streamlit run app.py

---

## Features
- Sentiment classification using BERT
- Real-time prediction
- Streamlit UI
- High accuracy (~90–95%)

---

## Files
- app.py → Streamlit interface
- sentiment_analysis_bert.py → Model logic
- test_sentiment.py → Script for testing predictions
- style.css → UI styling

---

## Note
This project was developed as part of MCA coursework.

---

## Future Improvements
- Multi-class sentiment analysis (positive, negative, neutral)
- Social media sentiment analysis
- Deployment on cloud
