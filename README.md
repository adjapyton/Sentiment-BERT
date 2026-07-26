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
## Modèle utilisé

Le projet utilise un modèle BERT (`bert-base-uncased`) entraîné avec PyTorch pour classifier les sentiments des textes en trois catégories :
- négatif
- neutre
- positif

L'entraînement est réalisé avec un fine-tuning de BERT sur le jeu de données Corona_NLP.

Le modèle BERT entraîné n'est pas inclus dans le dépôt GitHub en raison de sa taille importante. Il est généré automatiquement après l'exécution du script `train.py`.