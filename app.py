import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# ---------------- LOAD CUSTOM CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class='header'>
    <h1>BERT Sentiment Analyzer</h1>
    <p>Analyze emotions in text instantly using the power of BERT</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---", unsafe_allow_html=True)

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    model_path = "./downloads/bert_sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- EXAMPLES ----------------
st.write("**Try an example:**")
examples = [
    "The movie was absolutely amazing, I loved every moment!",
    "The food was terrible and the service was even worse.",
    "I feel so happy and motivated today!",
    "It was an average experience, nothing special."
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex[:25] + ("..." if len(ex) > 25 else ""), key=f"ex{i}"):

        st.session_state["user_text"] = ex

# ---------------- INPUT ----------------
default_text = st.session_state.get("user_text", "")
user_input = st.text_area("Enter your text:", height=150, value=default_text,
                          placeholder="Type or paste your text here...")

# ---------------- PREDICTION ----------------
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(1)
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                probs = torch.softmax(logits, dim=1).flatten().tolist()

            pos_conf, neg_conf = probs[1]*100, probs[0]*100
            sentiment = "Positive" if prediction == 1 else "Negative"
            color = "#3DDC84" if prediction == 1 else "#FF5C5C"
            conf = max(pos_conf, neg_conf)

        # ---------------- RESULT ----------------
        st.markdown(f"""
        <div class='result-box' style='border-left: 6px solid {color};'>
            <p class='result-text'><strong>Sentiment:</strong> {sentiment}</p>
            <p class='confidence'>Confidence: {conf:.2f}%</p>
            <div class='progress'>
                <div class='progress-bar' style='background:{color}; width:{conf}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
<hr>
<p><strong>Project by:</strong> Sabeena Kachary, Ruksar Khatun, Megha Saha</p>
<p>Built with ❤️ using BERT and Streamlit</p>
</div>
""", unsafe_allow_html=True)
