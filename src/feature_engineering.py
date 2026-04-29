"""
Jour 2 - TF-IDF + Cosine Similarity
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os


class FeatureExtractor:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, min_df=1, stop_words="english"
        )
        self.tfidf_matrix = None
        self.questions = []
        self.est_entraine = False

    def entrainer(self, questions: list):
        self.questions = questions
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        self.est_entraine = True
        print(f"[INFO] TF-IDF entraîné : {len(questions)} questions, {len(self.vectorizer.vocabulary_)} termes.")

    def vectoriser_requete(self, q: str):
        return self.vectorizer.transform([q])

    def calculer_similarites(self, vec) -> np.ndarray:
        return cosine_similarity(vec, self.tfidf_matrix).flatten()

    def top_k(self, scores: np.ndarray, k=3):
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    def sauvegarder(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix, "questions": self.questions}, f)
        print(f"[INFO] Modèle sauvegardé → {path}")

    def charger(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.vectorizer, self.tfidf_matrix, self.questions = d["vectorizer"], d["matrix"], d["questions"]
        self.est_entraine = True
        print(f"[INFO] Modèle chargé ← {path}")
