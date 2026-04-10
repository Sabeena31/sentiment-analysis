from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your fine-tuned model
model_path = "./downloads/bert_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        return sentiment

# Try some examples
samples = [
    "I absolutely loved this movie! The acting was fantastic.",
    "It was a total waste of time, I hated it.",
    "Pretty average film, not too bad but not great either."
]

for s in samples:
    print(f"{s} → {predict_sentiment(s)}")
