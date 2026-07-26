# Analyse de sentiments avec BERT

## Description

Ce projet entraîne un modèle BERT avec PyTorch afin de classifier des tweets selon leur sentiment.

Les sentiments sont regroupés en trois classes :

- Positive
- Neutral
- Negative

## Jeu de données

Le jeu de données utilisé est **Corona_NLP_train.csv**.

Les colonnes utilisées sont :

- OriginalTweet
- Sentiment

Les sentiments "Extremely Positive" et "Extremely Negative" ont été fusionnés afin d'obtenir trois classes.

## Modèle

Le modèle utilisé est :

- bert-base-uncased

Bibliothèque :

- Transformers (Hugging Face)

## Entraînement

- Framework : PyTorch
- Optimiseur : AdamW
- Epochs : 1
- Batch size : 8

## Démonstration

Une interface Gradio permet de saisir un texte et d'obtenir le sentiment prédit.

## Structure du projet

```
Sentiment BERT/
│
├── DATA/
├── Saved_Model/
├── train.py
├── model.py
├── demo.py
├── requirements.txt
└── README.md
```