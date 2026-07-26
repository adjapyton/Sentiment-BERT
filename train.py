import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# ============================
# Charger le jeu de données
# ============================

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "DATA" / "Corona_NLP_train.csv"

df = pd.read_csv(csv_path, encoding="latin1")

# Garder uniquement les colonnes utiles
df = df[["OriginalTweet", "Sentiment"]]

# Transformer les 5 classes en 3 classes
df["Sentiment"] = df["Sentiment"].replace({
    "Extremely Negative": "Negative",
    "Extremely Positive": "Positive"
})
df = df.sample(n=2000, random_state=42)
print("Classes :")
print(df["Sentiment"].unique())

# Encoder les labels
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["Sentiment"])

print("\nCorrespondance des labels :")
for i, c in enumerate(encoder.classes_):
    print(i, "->", c)

# Découper en entraînement / test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["OriginalTweet"],
    df["label"],
    test_size=0.2,
    random_state=42
)

print("\nNombre d'exemples d'entraînement :", len(train_texts))
print("Nombre d'exemples de test :", len(test_texts))

# ============================
# Charger le tokenizer BERT
# ============================

print("\nChargement du tokenizer...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Tokenizer chargé avec succès !")

# Exemple de tokenisation
texte = train_texts.iloc[0]

encodage = tokenizer(
    texte,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print("\nDimensions :")
print("input_ids :", encodage["input_ids"].shape)
print("attention_mask :", encodage["attention_mask"].shape)
import torch
from torch.utils.data import Dataset, DataLoader
from model import get_model

class TweetDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
test_dataset = TweetDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Dataset créé avec succès.")
from torch.optim import AdamW
from tqdm import tqdm
import os

# Choisir CPU ou GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nAppareil utilisé : {device}")

# Charger le modèle
model = get_model()
model.to(device)

# Optimiseur
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 1

print("\nDébut de l'entraînement...")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress = tqdm(train_loader)

    for batch in progress:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        progress.set_description(
            f"Epoch {epoch+1}/{epochs} Loss:{loss.item():.4f}"
        )

    print(f"\nEpoch {epoch+1} terminée")
    print("Loss moyenne :", total_loss / len(train_loader))

# Sauvegarder le modèle
os.makedirs("Saved_Model", exist_ok=True)

model.save_pretrained("Saved_Model")
tokenizer.save_pretrained("Saved_Model")

print("\n==============================")
print("Entraînement terminé !")
print("Modèle sauvegardé dans Saved_Model/")
print("==============================")