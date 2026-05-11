"""
Jour 1 - Données : Chargement et nettoyage du dataset banking77
"""

import pandas as pd, re, unicodedata, os, sys

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

def verifier_dataset():
    manquants = [f for f in [TRAIN_CSV, TEST_CSV] if not os.path.exists(f)]
    if manquants:
        print("=" * 55)
        print("DATASET MANQUANT !")
        print("Lance d'abord :  python setup.py")
        print("=" * 55)
        sys.exit(1)


def normaliser_texte(texte: str) -> str:
    if not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^\w\s']", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def charger_banking77() -> pd.DataFrame:
    verifier_dataset()
    df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "category" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    print(f"[INFO] Dataset : {len(df)} entrées, {df['label'].nunique()} catégories.")
    return df


def construire_faq(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in df.groupby("label"):
        for _, row in grp.iterrows():
            rows.append({
                "question":      row["text"],
                "reponse":       label,
                "categorie":     label,
                "question_clean": normaliser_texte(row["text"])
            })
    faq = pd.DataFrame(rows).drop_duplicates(subset=["question_clean"]).reset_index(drop=True)
    print(f"[INFO] FAQ : {len(faq)} entrées, {faq['categorie'].nunique()} catégories.")
    return faq
