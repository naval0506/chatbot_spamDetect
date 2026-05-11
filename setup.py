"""
setup.py — À lancer UNE SEULE FOIS avant de démarrer l'app.

    python setup.py

Ce script :
  1. Installe les dépendances (pip)
  2. Télécharge le dataset banking77 depuis HuggingFace
  3. Convertit en CSV dans data/
  4. Entraîne le modèle TF-IDF et le sauvegarde dans data/
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

# ── Étape 1 : pip install ─────────────────────────────────────────────────────
print("=" * 55)
print("Étape 1/4 — Installation des dépendances...")
print("=" * 55)
pkgs = ["streamlit", "scikit-learn", "pandas", "numpy", "datasets"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)

# ── Étape 2 : Télécharger banking77 ──────────────────────────────────────────
print("\n" + "=" * 55)
print("Étape 2/5 — Téléchargement du dataset banking77...")
print("(~5 MB, nécessite internet)")
print("=" * 55)

from datasets import load_dataset
import pandas as pd

ds = load_dataset("PolyAI/banking77")

# Convertir en DataFrame
df_train = ds["train"].to_pandas()
df_test  = ds["test"].to_pandas()

# banking77 stocke le label comme entier → récupérer le nom de la classe
label_names = ds["train"].features["label"].names  # liste de 77 noms

df_train["label_name"] = df_train["label"].apply(lambda x: label_names[x])
df_test["label_name"]  = df_test["label"].apply(lambda x: label_names[x])

# Sauvegarder
train_path = os.path.join(DATA, "train.csv")
test_path  = os.path.join(DATA, "test.csv")

df_train_cleaned = df_train[["text", "label_name"]].rename(columns={"label_name": "label"})
df_test_cleaned  = df_test[["text",  "label_name"]].rename(columns={"label_name": "label"})

df_train_cleaned.to_csv(train_path, index=False)
df_test_cleaned.to_csv(test_path,  index=False)

print(f"[OK] train.csv : {len(df_train_cleaned)} lignes")
print(f"[OK] test.csv  : {len(df_test_cleaned)} lignes")

# ── Étape 3 : Construire la base FAQ + entraîner TF-IDF ──────────────────────
print("\n" + "=" * 55)
print("Étape 3/5 — Construction de la base FAQ...")
print("=" * 55)

sys.path.insert(0, os.path.join(BASE, "src"))
from data_preprocessing import charger_banking77, construire_faq
from feature_engineering import FeatureExtractor

df_raw = charger_banking77()
df_faq = construire_faq(df_raw)

print("\n" + "=" * 55)
print("Étape 4/5 — Entraînement du modèle TF-IDF...")
print("=" * 55)

ext = FeatureExtractor()
ext.entrainer(df_faq["question_clean"].tolist())
ext.sauvegarder(os.path.join(DATA, "tfidf_model.pkl"))

print("\n" + "=" * 55)
print("Étape 5/5 — Entraînement du classifieur d'intention...")
print("=" * 55)

from trainer import ModelTrainer
trainer = ModelTrainer()
trainer.train(df_train_cleaned, df_test_cleaned)
trainer.save(os.path.join(DATA, "classifier_model.pkl"))

print("\n" + "=" * 55)
print("✅  Setup terminé !")
print("Lance maintenant :  streamlit run app.py")
print("=" * 55)
