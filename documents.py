"""
Fichier de configuration des documents pour le chatbot RAG

INSTRUCTIONS:
1. Modifie la liste DOCUMENTS ci-dessous avec tes propres textes
2. Chaque élément de la liste = 1 document (2-5 phrases recommandé)
3. Choisis un thème cohérent pour tous les documents
4. Plus tu as de documents, meilleure sera la couverture du sujet

EXEMPLES DE THÈMES:
- Cours de NLP/IA (comme ci-dessous)
- Médecine/Santé
- Histoire
- Littérature
- Sciences
- Économie
- Droit
- Sport
- Etc.
"""

# ============================================================
# 📚 BASE DOCUMENTAIRE - À PERSONNALISER SELON TON SUJET
# ============================================================

DOCUMENTS = [
    # Document 1
    """L'intelligence artificielle est une discipline scientifique qui vise à créer 
    des systèmes capables de réaliser des tâches nécessitant normalement l'intelligence 
    humaine. Elle englobe plusieurs domaines comme l'apprentissage automatique, 
    le traitement du langage naturel et la vision par ordinateur.""",
    
    # Document 2
    """Le machine learning est une branche de l'IA qui permet aux ordinateurs 
    d'apprendre à partir de données sans être explicitement programmés. 
    Les algorithmes de ML identifient des patterns dans les données pour faire 
    des prédictions ou prendre des décisions.""",
    
    # Document 3
    """Les réseaux de neurones artificiels sont inspirés du fonctionnement du cerveau 
    humain. Ils sont composés de couches de neurones interconnectés qui transforment 
    progressivement les données d'entrée pour produire une sortie.""",
    
    # Document 4
    """Le deep learning utilise des réseaux de neurones profonds avec de nombreuses 
    couches cachées. Cette approche a révolutionné des domaines comme la reconnaissance 
    d'images, la traduction automatique et la génération de texte.""",
    
    # Document 5
    """Le traitement du langage naturel (NLP) permet aux machines de comprendre et 
    générer du langage humain. Il inclut des tâches comme l'analyse de sentiment, 
    la traduction, la génération de texte et la réponse aux questions.""",
    
    # Document 6
    """Les Transformers sont une architecture de réseau de neurones basée sur 
    l'attention. Ils ont révolutionné le NLP en permettant de traiter des séquences 
    longues efficacement. Des modèles comme BERT et GPT utilisent cette architecture.""",
    
    # Document 7
    """Le RAG (Retrieval-Augmented Generation) combine la recherche d'information 
    et la génération de texte. Le système récupère d'abord des documents pertinents 
    puis génère une réponse basée sur ces documents, ce qui améliore la fiabilité.""",
    
    # Document 8
    """Les embeddings sont des représentations vectorielles denses du texte. 
    Ils capturent le sens sémantique des mots ou phrases, permettant de mesurer 
    la similarité entre textes de manière numérique.""",
    
    # Document 9
    """FAISS (Facebook AI Similarity Search) est une bibliothèque optimisée pour 
    la recherche de similarité dans de grands ensembles de vecteurs. Elle est 
    essentielle pour construire des systèmes de retrieval efficaces.""",
    
    # Document 10
    """Un chatbot intelligent combine plusieurs technologies : compréhension du 
    langage, gestion du contexte conversationnel, récupération d'information 
    et génération de réponses cohérentes et pertinentes.""",
    
    # Document 11
    """Le fine-tuning consiste à adapter un modèle pré-entraîné à une tâche 
    spécifique en l'entraînant sur un dataset ciblé. Cela permet d'obtenir 
    de meilleures performances qu'un modèle générique.""",
    
    # Document 12
    """L'attention est un mécanisme qui permet au modèle de se concentrer sur 
    les parties pertinentes de l'entrée. C'est la base des Transformers et 
    explique leur efficacité sur les tâches de séquence.""",
    
    # Document 13
    """Le prompt engineering consiste à formuler soigneusement les instructions 
    données à un modèle de langage pour obtenir les meilleurs résultats. 
    C'est devenu une compétence essentielle avec les grands modèles.""",
    
    # Document 14
    """Les modèles multimodaux peuvent traiter plusieurs types de données : 
    texte, images, audio. Ils ouvrent la voie à des applications plus riches 
    combinant différentes modalités d'information.""",
    
    # Document 15
    """L'éthique de l'IA soulève des questions importantes : biais algorithmiques, 
    confidentialité des données, transparence des décisions, impact sociétal. 
    Ces aspects doivent être considérés lors du développement de systèmes IA."""
]

# ============================================================
# 🎨 EXEMPLES D'AUTRES THÉMATIQUES
# ============================================================

# Décommente une section ci-dessous pour utiliser un autre thème

# --- MÉDECINE / SANTÉ ---
"""
DOCUMENTS_MEDECINE = [
    "Le diabète de type 2 est une maladie chronique caractérisée par une résistance 
    à l'insuline. Il peut être géré par l'alimentation, l'exercice et des médicaments.",
    
    "L'hypertension artérielle est souvent appelée 'tueur silencieux' car elle 
    présente peu de symptômes. Un suivi régulier de la tension est essentiel.",
    
    # Ajoute tes documents ici...
]
"""

# --- HISTOIRE ---
"""
DOCUMENTS_HISTOIRE = [
    "La Révolution française de 1789 a marqué la fin de la monarchie absolue en France. 
    Elle a introduit les concepts de liberté, égalité et fraternité.",
    
    "La Première Guerre mondiale (1914-1918) fut un conflit d'une ampleur sans précédent. 
    Elle a redessiné la carte de l'Europe et changé le cours du XXe siècle.",
    
    # Ajoute tes documents ici...
]
"""

# --- LITTÉRATURE ---
"""
DOCUMENTS_LITTERATURE = [
    "Le romantisme est un mouvement littéraire du XIXe siècle qui valorise 
    l'émotion, l'imagination et la nature. Victor Hugo en est un représentant majeur.",
    
    "L'existentialisme explore la liberté humaine et la responsabilité individuelle. 
    Sartre et Camus sont des figures clés de ce courant philosophique et littéraire.",
    
    # Ajoute tes documents ici...
]
"""

# ============================================================
# 📝 CONSEILS POUR CRÉER DE BONS DOCUMENTS
# ============================================================
"""
✅ BONNES PRATIQUES:
- 3-5 phrases par document (ni trop court, ni trop long)
- Texte clair et bien structuré
- Informations factuelles et précises
- Vocabulaire cohérent dans tous les documents
- Couvrir différents aspects du même thème

❌ À ÉVITER:
- Textes trop longs (> 10 phrases)
- Textes trop courts (1 phrase)
- Informations contradictoires entre documents
- Langage trop technique sans contexte
- Répétitions inutiles

💡 ASTUCE:
Tu peux aussi charger des documents depuis des fichiers texte:

def load_documents_from_files():
    documents = []
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"]:
        with open(file, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents
"""

# ============================================================
# 🔧 FONCTION D'EXPORT (ne pas modifier)
# ============================================================

def get_documents():
    """Retourne la liste des documents configurés"""
    return DOCUMENTS

def get_document_count():
    """Retourne le nombre de documents"""
    return len(DOCUMENTS)

def get_document_stats():
    """Retourne des statistiques sur les documents"""
    total_words = sum(len(doc.split()) for doc in DOCUMENTS)
    avg_words = total_words / len(DOCUMENTS)
    
    return {
        "count": len(DOCUMENTS),
        "total_words": total_words,
        "avg_words_per_doc": round(avg_words, 1)
    }