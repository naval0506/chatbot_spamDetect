"""
Jour 3-4 - Moteur de matching
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import charger_banking77, construire_faq, normaliser_texte
from feature_engineering import FeatureExtractor

BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE, "data", "tfidf_model.pkl")
CLASSIFIER_PATH = os.path.join(BASE, "data", "classifier_model.pkl")

DIRECT = {
    "greet":  ({"hello","hi","hey","good morning","good afternoon","howdy"},
               "Hello! I'm your banking assistant. How can I help you today? 🏦"),
    "bye":    ({"bye","goodbye","see you","ciao","au revoir"},
               "Goodbye! Don't hesitate to come back if you need help. 👋"),
    "thanks": ({"thanks","thank you","thank you so much","many thanks","cheers","merci"},
               "You're welcome! Is there anything else I can help you with?"),
}


class ChatbotBanking:
    SEUIL_SIM  = 0.08
    SEUIL_CLF  = 0.25

    def __init__(self):
        self.df    = None
        self.ext   = FeatureExtractor()
        self.clf   = None
        self.pret  = False

    def initialiser(self):
        from data_preprocessing import REPONSES
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

    def repondre(self, msg: str) -> dict:
        from data_preprocessing import REPONSES
        msg = msg.strip()
        low = msg.lower()

        for intent, (words, rep) in DIRECT.items():
            if any(w in low for w in words):
                return {"reponse": rep, "source": intent, "score": 1.0, "categorie": None,
                        "question_matchee": None, "suggestions": []}

        clean  = normaliser_texte(msg)
        
        # Mode 1 : Classifieur (si disponible)
        if self.clf:
            probs = self.clf.predict_proba([clean])[0]
            idx_best = probs.argmax()
            score_clf = probs[idx_best]
            label = self.clf.classes_[idx_best]
            
            if score_clf >= self.SEUIL_CLF:
                rep = REPONSES.get(label, f"I have information about {label.replace('_', ' ')}. How can I help?")
                return {"reponse": rep, "source": "classifier", "score": float(score_clf), 
                        "categorie": label, "question_matchee": f"Predicted intent: {label}", "suggestions": []}

        # Mode 2 : Similarité (Fallback)
        vec    = self.ext.vectoriser_requete(clean)
        scores = self.ext.calculer_similarites(vec)
        top    = self.ext.top_k(scores, k=3)
        idx, score = top[0]

        if score >= self.SEUIL_SIM:
            row  = self.df.iloc[idx]
            sugg = [self.df.iloc[i]["question"] for i, s in top[1:] if s > 0.05]
            return {"reponse": row["reponse"], "source": "similarity", "score": float(score),
                    "categorie": row["categorie"], "question_matchee": row["question"],
                    "suggestions": sugg}

        return {"reponse": "I'm sorry, I couldn't find a precise answer. Please try rephrasing or contact support.",
                "source": "not_found", "score": float(score), "categorie": None,
                "question_matchee": None, "suggestions": []}

    def categories(self):
        return sorted(self.df["categorie"].unique()) if self.df is not None else []

    def questions_par_cat(self, cat, n=8):
        if self.df is None: return []
        return self.df[self.df["categorie"] == cat]["question"].head(n).tolist()
