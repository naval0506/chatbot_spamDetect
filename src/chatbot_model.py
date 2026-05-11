"""
Jour 3-4 - Moteur de matching
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import charger_banking77, construire_faq, normaliser_texte
from feature_engineering import FeatureExtractor
from faq_responses import get_faq_response

BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE, "data", "tfidf_model.pkl")
CLASSIFIER_PATH = os.path.join(BASE, "data", "classifier_model.pkl")

class ChatbotBanking:
    SEUIL_SIM  = 0.08
    SEUIL_CLF  = 0.25
    SEUIL_SIM_FORT = 0.18
    SEUIL_CLF_FORT = 0.55

    def __init__(self):
        self.df    = None
        self.ext   = FeatureExtractor()
        self.clf   = None
        self.pret  = False

    def initialiser(self):
        print("[Bot] Chargement...")
        df_raw   = charger_banking77()
        self.df  = construire_faq(df_raw)
        
        # Load similarity model
        if os.path.exists(MODEL_PATH):
            self.ext.charger(MODEL_PATH)
        else:
            print("[Bot] Modèle TF-IDF non trouvé, entraînement rapide...")
            self.ext.entrainer(self.df["question_clean"].tolist())
            self.ext.sauvegarder(MODEL_PATH)
            
        # Load classifier model if exists
        if os.path.exists(CLASSIFIER_PATH):
            import pickle
            with open(CLASSIFIER_PATH, 'rb') as f:
                data = pickle.load(f)
                self.clf = data['pipeline']
            print(f"[Bot] Classifieur chargé (Précision : {data.get('metrics', {}).get('precision', 0):.2%})")
            
        self.pret = True
        print(f"[Bot] Prêt — {len(self.df)} questions, {self.df['categorie'].nunique()} catégories.")

    def _label_lisible(self, label: str | None) -> str:
        if not label:
            return "banking"
        return label.replace("_", " ").strip().title()

    def _exemples_categorie(self, label: str, exclude: str | None = None, n: int = 3) -> list[str]:
        if self.df is None or not label:
            return []

        exemples = []
        rows = self.df[self.df["categorie"] == label]["question"].head(n + 2).tolist()
        for question in rows:
            if question != exclude:
                exemples.append(question)
            if len(exemples) == n:
                break
        return exemples

    def _reponse_dynamique(self, label: str, question_matchee: str | None, score: float, source: str) -> str:
        sujet = self._label_lisible(label)
        confidence = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
        answer = get_faq_response(label)

        lignes = [
            answer,
            f"Detected topic: **{sujet}** ({confidence} confidence).",
        ]

        if question_matchee:
            lignes.append(f"The closest Banking77 example is: _{question_matchee}_")

        if source == "classifier":
            lignes.append("This answer comes from the intent classifier, so it is based on the wording of your message.")
        elif source == "similarity":
            lignes.append("This answer comes from semantic similarity with existing Banking77 questions.")
        else:
            lignes.append("I combined the classifier and the closest dataset example to choose this topic.")

        lignes.append(
            "To get a more precise result, include the status, amount, country/currency, card type, or date if it applies."
        )
        return "\n\n".join(lignes)

    def _resultat(self, label: str, score: float, source: str, question_matchee: str | None,
                  suggestions: list[str] | None = None) -> dict:
        return {
            "reponse": self._reponse_dynamique(label, question_matchee, score, source),
            "source": source,
            "score": float(score),
            "categorie": label,
            "question_matchee": question_matchee,
            "suggestions": suggestions or self._exemples_categorie(label, question_matchee),
        }

    def repondre(self, msg: str) -> dict:
        msg = msg.strip()
        clean  = normaliser_texte(msg)

        # Similarité toujours calculée: elle sert de garde-fou lorsque le
        # classifieur est trop confiant sur une catégorie voisine.
        vec = self.ext.vectoriser_requete(clean)
        scores = self.ext.calculer_similarites(vec)
        top = self.ext.top_k(scores, k=5)
        idx_sim, score_sim = top[0]
        row_sim = self.df.iloc[idx_sim]
        label_sim = row_sim["categorie"]
        question_sim = row_sim["question"]
        suggestions = [self.df.iloc[i]["question"] for i, s in top[1:] if s > 0.05]

        label_clf = None
        score_clf = 0.0

        if self.clf:
            probs = self.clf.predict_proba([clean])[0]
            idx_best = probs.argmax()
            score_clf = float(probs[idx_best])
            label_clf = self.clf.classes_[idx_best]

        if label_clf == label_sim and score_clf >= self.SEUIL_CLF:
            score = max(score_clf, score_sim)
            return self._resultat(label_clf, score, "hybrid", question_sim, suggestions)

        if score_sim >= self.SEUIL_SIM_FORT and score_clf < self.SEUIL_CLF_FORT:
            return self._resultat(label_sim, score_sim, "similarity", question_sim, suggestions)

        if label_clf and score_clf >= self.SEUIL_CLF:
            question = f"Predicted intent: {self._label_lisible(label_clf)}"
            return self._resultat(label_clf, score_clf, "classifier", question)

        if score_sim >= self.SEUIL_SIM:
            return self._resultat(label_sim, score_sim, "similarity", question_sim, suggestions)

        return {"reponse": "I could not identify a reliable banking topic. Try a more specific sentence with the card, transfer, top-up, verification, or payment detail.",
                "source": "not_found", "score": float(max(score_sim, score_clf)), "categorie": None,
                "question_matchee": None, "suggestions": []}

    def categories(self):
        return sorted(self.df["categorie"].unique()) if self.df is not None else []

    def questions_par_cat(self, cat, n=8):
        if self.df is None: return []
        return self.df[self.df["categorie"] == cat]["question"].head(n).tolist()
