# Rapport de projet - Sujet 3

## Chatbot intelligent pour assistance utilisateur avec deploiement

### 1. Contexte

Les entreprises recoivent beaucoup de questions repetitives de la part des utilisateurs : carte bancaire perdue, virement en attente, paiement refuse, probleme de verification, frais, retrait, etc. Repondre manuellement a toutes ces demandes prend du temps et peut ralentir le service client.

L'objectif de ce projet est donc de realiser un chatbot capable de comprendre une question utilisateur en langage naturel, de retrouver l'intention principale, puis de proposer une reponse pertinente a partir d'une base FAQ.

Le domaine choisi est l'assistance bancaire. L'application finale s'appelle **Bannki** et elle est accessible via une interface web Streamlit.

### 2. Objectif du projet

Le projet repond au sujet suivant : **Chatbot intelligent pour assistance utilisateur avec deploiement**.

Les objectifs techniques sont :

- comprendre une question utilisateur ;
- fournir une reponse pertinente ;
- apprendre a partir d'un dataset FAQ ;
- utiliser TF-IDF et la similarite cosinus ;
- extraire l'intention utilisateur ;
- deployer le chatbot dans une interface web ;
- respecter la contrainte : **Python + scikit-learn**.

### 3. Problematique

La problematique traitee est :

**Comment concevoir un chatbot capable de comprendre le langage naturel et de repondre efficacement a une question utilisateur ?**

Pour y repondre, j'ai construit un pipeline simple et interpretable :

1. nettoyer les questions ;
2. transformer les textes en vecteurs TF-IDF ;
3. comparer la question utilisateur avec les questions connues ;
4. predire l'intention bancaire avec un classifieur ;
5. afficher le resultat dans une interface de chat.

### 4. Donnees utilisees - Jour 1

Le dataset utilise est **Banking77**, fourni par **PolyAI** et disponible sur HuggingFace :

https://huggingface.co/datasets/PolyAI/banking77

Ce dataset contient des questions reelles ou realistes de service client bancaire. Il est adapte au projet car il est deja organise comme une FAQ d'assistance utilisateur.

Caracteristiques du dataset :

| Element | Valeur |
|---|---:|
| Nom | Banking77 |
| Source | PolyAI / HuggingFace |
| Domaine | Assistance bancaire |
| Nombre total d'exemples charges | 13 083 |
| Nombre d'exemples train | 10 003 |
| Nombre d'exemples test | 3 080 |
| Nombre de categories | 77 |
| Colonnes conservees | `text`, `label` |

Exemples de categories :

- `card_arrival`
- `lost_or_stolen_card`
- `transfer_timing`
- `pending_top_up`
- `passcode_forgotten`
- `apple_pay_or_google_pay`
- `pin_blocked`

Les donnees sont telechargees par [setup.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/setup.py), puis sauvegardees dans :

- [data/train.csv](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/data/train.csv)
- [data/test.csv](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/data/test.csv)

### 5. Nettoyage du texte - Jour 1

Le nettoyage est implemente dans [src/data_preprocessing.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/data_preprocessing.py).

Les traitements appliques sont :

- passage en minuscules ;
- normalisation Unicode ;
- suppression des accents ;
- suppression de la ponctuation inutile ;
- remplacement des espaces multiples par un seul espace ;
- suppression des doublons dans la FAQ finale.

Exemple :

```text
Question originale : What can I do if my card still hasn't arrived after 2 weeks?
Question nettoyee  : what can i do if my card still hasn't arrived after 2 weeks
```

Apres nettoyage et suppression des doublons, la base FAQ contient **13 023 questions uniques** reparties sur **77 categories**.

### 6. Feature engineering - Jour 2

Pour representer les textes sous forme numerique, j'ai utilise **TF-IDF** avec `scikit-learn`.

Le fichier concerne est [src/feature_engineering.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/feature_engineering.py).

Le TF-IDF permet de donner un poids important aux mots utiles pour distinguer les intentions. Par exemple, dans le domaine bancaire, des mots comme `card`, `transfer`, `pin`, `cash`, `top up`, `refund` ou `identity` sont importants.

Configuration principale :

| Parametre | Valeur |
|---|---|
| Methode | `TfidfVectorizer` |
| Nombre maximum de termes | 10 000 |
| N-grams | 1 a 2 mots |
| Stop words | anglais |
| Similarite | cosinus |

Le modele TF-IDF entraine est sauvegarde dans :

[data/tfidf_model.pkl](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/data/tfidf_model.pkl)

### 7. Similarite cosinus - Jour 2

Apres vectorisation, la question utilisateur est comparee a toutes les questions de la FAQ avec la **similarite cosinus**.

Principe :

1. l'utilisateur saisit une question ;
2. la question est nettoyee ;
3. elle est transformee en vecteur TF-IDF ;
4. le systeme calcule la similarite avec les questions connues ;
5. la question la plus proche est recuperee.

Cette methode permet de faire du **matching question/reponse**. Elle est utile quand la question utilisateur ressemble a une question deja presente dans le dataset.

### 8. Modelisation - Jour 3 et Jour 4

La logique principale du chatbot est dans :

[src/chatbot_model.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/chatbot_model.py)

Le chatbot utilise deux sources de decision :

| Source | Role |
|---|---|
| Similarite cosinus | Retrouver la question FAQ la plus proche |
| Classifieur d'intention | Predire directement la categorie de la question |

Le systeme combine les deux resultats :

- si le classifieur et la similarite trouvent la meme categorie, la reponse est consideree comme plus fiable ;
- si la similarite est forte, le chatbot utilise la question la plus proche ;
- si le classifieur est confiant, il utilise l'intention predite ;
- si aucun score n'est suffisant, le chatbot demande une question plus precise.

Des seuils sont utilises pour eviter de repondre n'importe quoi quand le modele n'est pas assez confiant :

| Seuil | Role |
|---|---|
| `SEUIL_SIM` | score minimum pour accepter une similarite FAQ |
| `SEUIL_CLF` | score minimum pour accepter une prediction du classifieur |
| `SEUIL_SIM_FORT` | similarite assez forte pour privilegier le matching |
| `SEUIL_CLF_FORT` | confiance forte du classifieur |

Ces seuils se trouvent dans `ChatbotBanking`. Ils permettent de limiter les mauvaises reponses et de retourner un message de clarification quand la question est trop vague.

### 9. Extraction d'intention - Jour 4

L'extraction d'intention est realisee avec un modele `scikit-learn` dans :

[src/trainer.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/trainer.py)

Le modele utilise :

- TF-IDF par mots ;
- TF-IDF par caracteres ;
- `FeatureUnion` pour combiner les deux representations ;
- `LinearSVC` pour la classification ;
- `CalibratedClassifierCV` pour obtenir un score de confiance avec `predict_proba`.

Le classifieur predit une intention parmi les **77 categories Banking77**.

Le modele entraine est sauvegarde dans :

[data/classifier_model.pkl](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/data/classifier_model.pkl)

La sauvegarde du modele est importante : elle evite de re-entrainer le modele a chaque lancement de l'application. Au demarrage, le chatbot charge directement :

- `data/tfidf_model.pkl` pour le matching par similarite ;
- `data/classifier_model.pkl` pour l'extraction d'intention.

Resultats du modele sur le jeu de test :

| Metrique | Score |
|---|---:|
| Accuracy | 0.9107 |
| Precision | 0.9142 |
| Recall | 0.9107 |
| F1-score | 0.9110 |

#### Bonnes pratiques d'entrainement et de validation

Pour suivre une demarche correcte de modele IA, j'ai separe les donnees d'entrainement et de test. Le dataset Banking77 fournit deja deux parties distinctes :

| Jeu de donnees | Role | Nombre d'exemples |
|---|---|---:|
| `train.csv` | Donnees utilisees pour apprendre le modele | 10 003 |
| `test.csv` | Donnees gardees pour evaluer le modele apres entrainement | 3 080 |

Le modele est entraine uniquement sur `train.csv`. Le fichier `test.csv` n'est pas utilise pendant l'apprentissage. Il sert seulement a mesurer les performances finales. Cela evite une fuite de donnees et permet d'avoir une evaluation plus fiable.

Le pipeline d'entrainement est le suivant :

1. charger `data/train.csv` et `data/test.csv` ;
2. prendre la colonne `text` comme entree du modele ;
3. prendre la colonne `label` comme intention a predire ;
4. transformer les textes avec TF-IDF ;
5. entrainer le classifieur `LinearSVC` calibre ;
6. predire les labels du jeu de test ;
7. comparer les predictions avec les vraies categories ;
8. calculer les metriques.

Les metriques utilisees sont :

| Metrique | Role |
|---|---|
| Accuracy | Pourcentage total de bonnes predictions |
| Precision | Fiabilite des predictions positives par categorie |
| Recall | Capacite a retrouver les exemples d'une categorie |
| F1-score | Equilibre entre precision et recall |

J'utilise les scores **weighted** pour `precision`, `recall` et `F1-score`, car il y a beaucoup de classes. Cette moyenne tient compte du nombre d'exemples par categorie.

Le script [src/trainer.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/trainer.py) genere aussi :

- un `classification_report` par categorie ;
- une matrice de confusion avec `confusion_matrix` ;
- les metriques globales ;
- les informations sauvegardees avec le modele.

Le rapport detaille d'entrainement est disponible dans :

[data/final_production_report.md](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/data/final_production_report.md)

Exemples d'interpretation :

- `passcode_forgotten` obtient un tres bon resultat, car les questions autour du mot de passe sont assez specifiques ;
- certaines categories proches comme les virements en attente, les transferts rates ou les problemes de solde peuvent etre plus difficiles a separer ;
- le score global autour de 91 % montre que le modele est satisfaisant pour une premiere version de chatbot FAQ.

Ces resultats montrent que le modele reconnait correctement la majorite des intentions du dataset tout en restant conforme a la contrainte `Python + scikit-learn`.

### 10. Reponse du chatbot

Le chatbot ne genere pas de texte libre comme un modele de deep learning. Il reste conforme au sujet et a la contrainte `Python + scikit-learn`.

Pour que l'application soit vraiment une assistante et pas seulement un detecteur de categorie, j'ai ajoute une base de reponses FAQ dans :

[src/faq_responses.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/faq_responses.py)

Ce fichier contient des reponses redigees pour les intentions Banking77. Par exemple, si l'intention detectee est `lost_or_stolen_card`, le chatbot repond qu'il faut bloquer la carte, signaler la perte ou le vol, puis demander une nouvelle carte.

La reponse contient :

- une reponse FAQ concrete liee a l'intention detectee ;
- le sujet detecte ;
- le niveau de confiance ;
- la question FAQ la plus proche ;
- la source de decision : `similarity`, `classifier` ou `hybrid` ;
- des exemples proches pour aider l'utilisateur a reformuler.

Exemple de sortie :

```text
Question : I lost my card, what should I do?
Reponse : If your card is lost or stolen, freeze it immediately in the app to stop new payments. Then report it as lost or stolen and order a replacement card.
Categorie detectee : lost_or_stolen_card
Source : hybrid
Score : 1.0
```

Le fonctionnement est donc le suivant :

1. le modele detecte l'intention ;
2. le chatbot cherche la reponse associee a cette intention dans `faq_responses.py` ;
3. il affiche la reponse utilisateur ;
4. il ajoute les informations techniques utiles : categorie, confiance, source et exemples proches.

Autres tests realises :

| Question utilisateur | Categorie detectee | Source | Score |
|---|---|---|---:|
| I lost my card, what should I do? | `lost_or_stolen_card` | hybrid | 1.000 |
| How long does a bank transfer take? | `balance_not_updated_after_bank_transfer` | hybrid | 0.833 |
| Why is my top up still pending? | `pending_top_up` | hybrid | 1.000 |
| I forgot my passcode | `passcode_forgotten` | hybrid | 1.000 |
| Can I use Apple Pay with my card? | `apple_pay_or_google_pay` | hybrid | 0.848 |

### 11. Fonctionnement complet du chatbot

Cette partie explique exactement ce qui se passe quand un utilisateur pose une question.

Exemple de question :

```text
I forgot my passcode
```

#### Etape 1 - Saisie utilisateur

L'utilisateur ecrit sa question dans l'interface Streamlit. Le champ de saisie est gere dans [app.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/app.py) avec :

```python
st.chat_input()
```

#### Etape 2 - Appel du moteur chatbot

Quand une question est envoyee, l'interface appelle la methode :

```python
bot.repondre(to_do)
```

Cette methode se trouve dans [src/chatbot_model.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/chatbot_model.py).

#### Etape 3 - Nettoyage

La phrase est nettoyee par `normaliser_texte()` dans [src/data_preprocessing.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/data_preprocessing.py).

Exemple :

```text
I forgot my passcode
```

devient :

```text
i forgot my passcode
```

#### Etape 4 - Matching avec TF-IDF et similarite cosinus

La question nettoyee est transformee en vecteur TF-IDF avec le modele sauvegarde dans `data/tfidf_model.pkl`. Ensuite, le programme calcule la similarite cosinus entre cette question et toutes les questions de la base FAQ.

Le but est de retrouver la question la plus proche dans Banking77.

#### Etape 5 - Prediction d'intention

En parallele, le classifieur charge depuis `data/classifier_model.pkl` predit l'intention utilisateur. Pour la question :

```text
I forgot my passcode
```

l'intention detectee est :

```text
passcode_forgotten
```

#### Etape 6 - Recuperation de la reponse FAQ

Le chatbot cherche ensuite la reponse correspondant a l'intention dans [src/faq_responses.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/src/faq_responses.py).

Pour `passcode_forgotten`, la reponse est :

```text
If you forgot your passcode, use the account recovery option in the app. You may need to verify your identity before setting a new passcode.
```

#### Etape 7 - Affichage dans Streamlit

L'interface affiche :

- la reponse FAQ ;
- la categorie detectee ;
- le score de confiance ;
- la source de decision : `similarity`, `classifier` ou `hybrid` ;
- la question Banking77 la plus proche ;
- des exemples similaires.

Ce fonctionnement permet de respecter les trois parties principales du sujet : comprehension de la question, reponse pertinente et interface web.

### 12. Deploiement Streamlit - Jour 5 et Jour 6

L'interface web est developpee avec **Streamlit** dans :

[app.py](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/app.py)

Fonctionnalites de l'interface :

- zone de chat ;
- historique de conversation ;
- exemples de questions predefinies ;
- compteur de questions ;
- taux de questions resolues ;
- liste des categories ;
- affichage de la categorie detectee ;
- affichage du score de confiance ;
- affichage des questions similaires.

Commande de lancement :

```bash
streamlit run app.py
```

Adresse locale :

```text
http://localhost:8501
```

### 13. Captures d'ecran de la demonstration

Les captures doivent etre placees dans le dossier :

[docs/screenshots](/home/valkely/Documents/nlpppp/mande/chatbot_banking77/chatbot_v3/docs/screenshots)

Captures prevues pour la soutenance :

| Capture | Description | Chemin conseille |
|---|---|---|
| Accueil | Interface Streamlit au lancement | `docs/screenshots/streamlit_accueil.png` |
| Question utilisateur | Exemple avec une question bancaire | `docs/screenshots/question_carte_perdue.png` |
| Resultat intention | Categorie, score et exemple proche | `docs/screenshots/resultat_intention.png` |
| Categories | Filtrage des categories dans la sidebar | `docs/screenshots/categories.png` |

Une tentative de capture automatique a ete faite avec Firefox headless, mais l'environnement avait deja une session Firefox ouverte, ce qui a bloque la generation automatique. Les captures peuvent donc etre prises manuellement apres lancement de l'application avec :

```bash
streamlit run app.py
```

Puis ouvrir :

```text
http://localhost:8501
```

### 14. Organisation du projet

```text
chatbot_v3/
├── app.py
├── setup.py
├── requirements.txt
├── README.md
├── rapport_projet.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── tfidf_model.pkl
│   └── classifier_model.pkl
└── src/
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── chatbot_model.py
    ├── faq_responses.py
    └── trainer.py
```

Role de chaque fichier :

| Fichier | Role dans le projet |
|---|---|
| `setup.py` | Telecharge Banking77, prepare les CSV et entraine les modeles |
| `app.py` | Lance l'interface web Streamlit et gere le chat |
| `requirements.txt` | Liste les dependances Python necessaires |
| `README.md` | Explique rapidement comment installer et lancer le projet |
| `rapport_projet.md` | Rapport complet du projet |
| `data/train.csv` | Donnees d'entrainement |
| `data/test.csv` | Donnees de test |
| `data/tfidf_model.pkl` | Modele TF-IDF sauvegarde |
| `data/classifier_model.pkl` | Classifieur d'intention sauvegarde |
| `src/data_preprocessing.py` | Nettoyage du texte et construction de la FAQ |
| `src/feature_engineering.py` | Vectorisation TF-IDF et similarite cosinus |
| `src/chatbot_model.py` | Moteur principal du chatbot |
| `src/faq_responses.py` | Reponses FAQ associees aux intentions |
| `src/trainer.py` | Entrainement et evaluation du classifieur |

### 15. Technologies utilisees

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| pandas | Chargement et manipulation des donnees |
| numpy | Operations numeriques |
| scikit-learn | TF-IDF, similarite, classification |
| Streamlit | Interface web du chatbot |
| HuggingFace datasets | Telechargement du dataset Banking77 |

### 16. Respect du cahier des charges

| Travail demande | Realisation dans le projet |
|---|---|
| Dataset FAQ | Banking77 |
| Nettoyage texte | Normalisation dans `data_preprocessing.py` |
| TF-IDF | `TfidfVectorizer` |
| Similarite cosinus | Matching FAQ dans `feature_engineering.py` |
| Matching question/reponse | `ChatbotBanking.repondre()` |
| Reponse pertinente | Reponses par intention dans `faq_responses.py` |
| Extraction d'intention | Classifieur `LinearSVC` |
| Interface web | Streamlit |
| Chat interface | `st.chat_input` et `st.chat_message` |
| Rapport | `rapport_projet.md` |
| Contrainte Python + scikit-learn | Respectee |

Le bonus embeddings n'a pas ete ajoute afin de rester simple et de ne pas depasser le perimetre principal du sujet.

### 17. Limites du projet

Le chatbot fonctionne bien pour les questions proches du domaine bancaire, mais il a quelques limites :

- il ne traite pas les demandes hors domaine bancaire ;
- il ne remplace pas un conseiller humain pour les cas sensibles ;
- il donne des reponses FAQ generales, pas des informations personnelles sur un vrai compte ;
- il depend de la qualite des exemples presents dans Banking77 ;
- certaines categories bancaires sont proches et peuvent etre confondues.

### 18. Ameliorations possibles

Des ameliorations peuvent etre ajoutees plus tard, sans changer la base du projet :

- ajouter une page d'administration pour enrichir la FAQ ;
- enrichir les reponses avec les procedures exactes d'une entreprise precise ;
- ajouter les embeddings en bonus ;
- exporter automatiquement les conversations anonymisees ;
- ajouter une evaluation utilisateur apres chaque reponse.

### 19. Conclusion

Ce projet realise un chatbot intelligent d'assistance bancaire conforme au Sujet 3. Il utilise un dataset FAQ reel, applique un nettoyage de texte, transforme les questions avec TF-IDF, calcule la similarite cosinus, predit l'intention utilisateur avec un classifieur `scikit-learn`, puis deploie le tout dans une interface web Streamlit.

La solution reste volontairement simple, interpretable et conforme aux contraintes demandees : **Python + scikit-learn**.
