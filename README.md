# Chatbot RAG avec Streamlit

##  Projet Académique

Chatbot intelligent utilisant un système RAG (Retrieval-Augmented Generation) avec interface Streamlit.

### Différences par rapport au notebook du prof :
- ✅ **Modèle d'embeddings différent** : `paraphrase-multilingual-mpnet-base-v2` (au lieu de all-MiniLM)
- ✅ **Modèle génératif différent** : `flan-t5-base` (au lieu de flan-t5-small)
- ✅ **Interface Streamlit** complète et interactive
- ✅ **100% gratuit et local** (pas d'API payante)

---

##  Prérequis

- Python 3.8 ou supérieur
- Windows 12
- VS Code
- Connexion internet (pour télécharger les modèles au premier lancement)

---

##  Installation

### 1. Créer un dossier pour le projet

```bash
mkdir chatbot_rag
cd chatbot_rag
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l'environnement virtuel

Sur Windows (PowerShell) :
```bash
venv\Scripts\Activate.ps1
```

Sur Windows (CMD) :
```bash
venv\Scripts\activate.bat
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

⏰ **Note** : L'installation peut prendre 5-10 minutes (PyTorch est lourd).

---

## 🎮 Lancement de l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

---

## Personnaliser la base documentaire

Pour adapter le chatbot à **ton propre sujet**, modifie la fonction `get_documents()` dans `app.py` :

```python
def get_documents():
    """Retourne la base documentaire (à adapter selon ton sujet)"""
    documents = [
        """Ton premier document sur ton sujet...""",
        
        """Ton deuxième document...""",
        
        """Etc..."""
    ]
    return documents
```

### Exemples de sujets possibles :
-  Cours de NLP / IA
-  Domaine médical
-  Support pédagogique d'une matière
-  Documentation d'entreprise
-  Articles scientifiques vulgarisés

---

##  Structure du projet

```
chatbot_rag/
│
├── app.py              # Application principale
├── requirements.txt    # Dépendances Python
├── README.md          # Ce fichier
│
└── venv/              # Environnement virtuel (créé automatiquement)
```

---

##  Fonctionnalités

### ✅ Implémentées
- Système RAG complet (Retrieval + Generation)
- Embeddings multilingues de haute qualité
- Index vectoriel FAISS pour recherche rapide
- Historique de conversation (2 derniers tours)
- Interface Streamlit intuitive
- Affichage des documents sources
- Scores de similarité visibles

### 🔮 Extensions possibles
- Ajout de fichiers PDF/TXT comme documents
- Découpage automatique en chunks
- Fine-tuning du modèle génératif
- Export de l'historique
- Analyse de sentiment des questions

---

##  Tester le chatbot

### Questions exemples :

1. **"C'est quoi l'intelligence artificielle ?"**
2. **"Explique le machine learning simplement"**
3. **"Quelle est la différence entre deep learning et ML ?"**
4. **"Comment fonctionne le RAG ?"**
5. **"Explique les Transformers en NLP"**

---

##  Résolution de problèmes

### Problème : Erreur d'import
```
Solution : Vérifier que l'environnement virtuel est activé
```

### Problème : Modèles trop lents
```
Solution : Utiliser flan-t5-small au lieu de flan-t5-base
Changer la ligne 41 dans app.py
```

### Problème : Manque de mémoire
```
Solution : Réduire max_length dans la fonction generate_answer
```

---

## 📊 Comparaison avec le notebook du prof

| Aspect | Notebook Prof | Notre Projet |
|--------|--------------|--------------|
| Modèle embeddings | all-MiniLM-L6-v2 | paraphrase-multilingual-mpnet-base-v2 |
| Modèle génération | flan-t5-small | flan-t5-base |
| Interface | Jupyter/Colab | Streamlit Web App |
| Index | NumPy ou FAISS | FAISS optimisé |
| Historique | Manuel | Automatique UI |

---

##  Concepts implémentés

- ✅ **Embeddings sémantiques** : Représentation vectorielle du texte
- ✅ **Similarité cosinus** : Mesure de proximité sémantique
- ✅ **FAISS** : Recherche efficace de voisins proches
- ✅ **RAG** : Récupération + Génération augmentée
- ✅ **Transformers** : Architecture moderne de NLP
- ✅ **Gestion du contexte** : Historique conversationnel

---

##  Développement

Le code est structuré de manière pédagogique avec :
- Commentaires explicatifs
- Fonctions modulaires
- Cache Streamlit pour optimisation
- Gestion propre de l'état de session

---

##  Licence

Projet académique - Libre d'utilisation pour l'apprentissage

---

##  Contribution

Pour améliorer le projet :
1. Ajouter plus de documents
2. Améliorer les prompts
3. Tester différents modèles
4. Optimiser les performances

---

**Bon développement ! 🚀**
