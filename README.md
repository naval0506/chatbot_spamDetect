# 🏦 Banking FAQ Chatbot — Banking77

Chatbot intelligent basé sur le dataset **Banking77** (13 083 questions, 77 catégories bancaires).

---

## ⚡ Démarrage rapide

```bash
# 1. Setup complet (télécharge les données + entraîne les modèles TF-IDF et LinearSVC)
python setup.py

# 2. Lancer l'interface web (Chatbot)
streamlit run app.py
```

---

## 📋 Ce que fait `setup.py`

| Étape | Action |
|-------|--------|
| 1 | `pip install` des dépendances |
| 2 | Télécharge banking77 depuis HuggingFace (~5 MB) |
| 3 | Convertit et nettoie les données (`data/train.csv`, `test.csv`) |
| 4 | Entraîne le modèle TF-IDF → `data/tfidf_model.pkl` |
| 5 | Entraîne le classifieur d'intention → `data/classifier_model.pkl` |

---

## 🏗️ Structure du projet

```text
chatbot_v3/
├── setup.py               ← LANCER EN PREMIER (Prépare les données et les modèles)
├── app.py                 ← Interface Streamlit (Chatbot)
├── requirements.txt       ← Dépendances
├── data/                  ← Généré par setup.py
│   ├── train.csv
│   ├── test.csv
│   ├── tfidf_model.pkl
│   └── classifier_model.pkl
└── src/
    ├── data_preprocessing.py  ← Jour 1 : Nettoyage texte
    ├── feature_engineering.py ← Jour 2 : TF-IDF + Similarité cosinus
    ├── chatbot_model.py       ← Jour 3-4 : Moteur de matching
    └── trainer.py             ← Jour 4 : Extraction d'intentions
```

---

## 📊 Dataset Banking77

- **Source** : [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **13 083 questions** réelles de service client bancaire
- **77 catégories** : `lost_or_stolen_card`, `transfer_timing`, `pin_blocked`...

---

## 🔬 Pipeline technique

```text
Question utilisateur
  │
  ▼
Normalisation (lowercase, accents, regex)
  │
  ▼
TF-IDF Vectorisation (10k features, bigrams)
  │
  ├── Mode 1 : Classifieur (LinearSVC)
  │            → Prédit l'intention avec score de confiance
  │
  └── Mode 2 : Similarité cosinus (fallback)
               → Cherche la question FAQ la plus proche
  │
  ▼
Réponse + catégorie + score de confiance + suggestions
```

---

## 🛠️ Technologies utilisées

| Outil | Usage |
|-------|-------|
| **Python 3.11** | Langage principal |
| **scikit-learn** | TF-IDF, LinearSVC, métriques |
| **pandas / numpy** | Manipulation des données |
| **Streamlit** | Interface web interactive |

---

## 📝 Contraintes respectées (Sujet 3)

- ✅ Python + scikit-learn uniquement
- ✅ Dataset FAQ réel (Banking77)
- ✅ Nettoyage de texte
- ✅ Feature engineering : TF-IDF + Similarité cosinus
- ✅ Modélisation : Matching question/réponse
- ✅ Extraction d'intention utilisateur (Classifieur)
- ✅ Déploiement : Interface Streamlit
