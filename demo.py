import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification

# Charger le tokenizer et le modèle
tokenizer = BertTokenizer.from_pretrained("Saved_Model")
model = BertForSequenceClassification.from_pretrained("Saved_Model")

model.eval()

labels = ["Negative", "Neutral", "Positive"]


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return labels[prediction]


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Saisissez une phrase..."),
    outputs=gr.Label(),
    title="Analyse des sentiments avec BERT",
    description="Saisissez une phrase pour en prédire le sentiment."
)

demo.launch()