# 🏦 Banking FAQ Chatbot — Banking77

Chatbot intelligent basé sur le dataset **Banking77** (13 083 questions, 77 catégories bancaires).

---

## ⚡ Démarrage rapide

```bash
# 1. Setup complet (installe tout + télécharge le dataset + entraîne le modèle TF-IDF)
python setup.py

# 2. Lancer l'interface web
streamlit run app.py
```

---

## 📋 Ce que fait `setup.py`

| Étape | Action |
|-------|--------|
| 1 | `pip install` de toutes les dépendances |
| 2 | Télécharge banking77 depuis HuggingFace (~5 MB) |
| 3 | Convertit en `data/train.csv` + `data/test.csv` |
| 4 | Entraîne le modèle TF-IDF → `data/tfidf_model.pkl` |

---

## 🏗️ Structure du projet

```
chatbot_v3/
├── setup.py               ← LANCER EN PREMIER
├── app.py                 ← Interface Streamlit (3 pages)
├── requirements.txt
├── Dockerfile             ← Conteneurisation
├── docker-compose.yml
├── test_trainer.py        ← Tests unitaires
├── data/                  ← Généré par setup.py
│   ├── train.csv
│   ├── test.csv
│   ├── tfidf_model.pkl
│   ├── classifier_model.pkl   (généré après entraînement)
│   └── training_report.md     (généré après export)
└── src/
    ├── data_preprocessing.py  ← Jour 1 : Nettoyage texte
    ├── feature_engineering.py ← Jour 2 : TF-IDF + Similarité cosinus
    ├── chatbot_model.py       ← Jour 3-4 : Moteur de matching + Extraction d'intentions
    └── trainer.py             ← Jour 3-4 : Entraînement + Validation du modèle
```

---

## 📊 Dataset Banking77

- **Source** : [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **13 083 questions** réelles de service client bancaire
- **77 catégories** : `lost_or_stolen_card`, `transfer_timing`, `pin_blocked`, `exchange_rate`…
- **Licence** : CC BY 4.0 (gratuit)

---

## 🔬 Pipeline technique

```
Question utilisateur
  │
  ▼
Normalisation (lowercase, accents, regex)
  │
  ▼
TF-IDF Vectorisation (10k features, bigrams)
  │
  ├── Mode 1 : Classifieur (Logistic Regression multinomiale)
  │            → Prédit l'intention avec score de confiance
  │
  └── Mode 2 : Similarité cosinus (fallback)
               → Cherche la question FAQ la plus proche
  │
  ▼
Réponse + catégorie + score de confiance + suggestions
```

---

## 🔍 Validation du modèle

Le projet inclut une validation rigoureuse :

| Méthode | Description |
|---------|-------------|
| **Train/Test Split** | Évaluation sur le jeu de test séparé |
| **Cross-Validation k-fold** | Validation stratifiée (2 à 10 folds) |
| **Métriques globales** | Accuracy, Precision, Recall, F1-Score (weighted) |
| **Métriques par classe** | Precision/Recall/F1 pour chaque catégorie |
| **Matrice de confusion** | Heatmap interactive (Plotly) |
| **Rapport exportable** | Rapport Markdown téléchargeable |

### Lancer la validation depuis l'interface

1. Aller sur la page **⚙️ Model Training** → cliquer **🚀 Start Training**
2. Aller sur la page **🔍 Model Validation** → choisir le nombre de folds → cliquer **▶️ Run Cross-Validation**
3. Consulter les résultats : métriques par fold, catégories faibles, rapport détaillé

---

## 🐳 Déploiement Docker

```bash
# Build et lancer
docker compose up --build

# L'app est accessible sur http://localhost:8501
```

---

## 🛠️ Technologies utilisées

| Outil | Usage |
|-------|-------|
| **Python 3.11** | Langage principal |
| **scikit-learn** | TF-IDF, Logistic Regression, métriques |
| **pandas / numpy** | Manipulation des données |
| **Streamlit** | Interface web interactive |
| **Plotly** | Visualisations interactives |
| **Docker** | Conteneurisation et déploiement |

---

## 📝 Contraintes respectées

- ✅ Python + scikit-learn uniquement (pas de deep learning)
- ✅ Dataset FAQ réel (Banking77)
- ✅ Nettoyage de texte (normalisation Unicode, regex)
- ✅ Feature engineering : TF-IDF + Similarité cosinus
- ✅ Modélisation : Matching question/réponse + Classification
- ✅ Extraction d'intention utilisateur
- ✅ Déploiement : Interface Streamlit + Docker
- ✅ Validation complète du modèle (CV, métriques, rapports)
