"""
Chatbot RAG avec Streamlit
Utilise paraphrase-multilingual-mpnet-base-v2 pour les embeddings
et google/flan-t5-base pour la génération
"""

import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

# Configuration de la page
st.set_page_config(
    page_title="Chatbot RAG Pédagogique",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# INITIALISATION DES MODÈLES (avec cache)
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embeddings - DIFFÉRENT du notebook prof"""
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    return SentenceTransformer(model_name)

@st.cache_resource
def load_generation_model():
    """Charge le modèle de génération - VERSION DIFFÉRENTE"""
    model_name = "google/flan-t5-base"  # Plus gros que small
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model, device

# ============================================================
# BASE DOCUMENTAIRE - Importée depuis documents.py
# ============================================================

try:
    from documents import get_documents, get_document_stats
    USE_EXTERNAL_DOCS = True
except ImportError:
    USE_EXTERNAL_DOCS = False
    
    def get_documents():
        """Base documentaire par défaut (fallback)"""
        return [
            """L'intelligence artificielle est une discipline scientifique qui vise à créer 
            des systèmes capables de réaliser des tâches nécessitant normalement l'intelligence 
            humaine. Elle englobe plusieurs domaines comme l'apprentissage automatique, 
            le traitement du langage naturel et la vision par ordinateur.""",
            
            """Le machine learning est une branche de l'IA qui permet aux ordinateurs 
            d'apprendre à partir de données sans être explicitement programmés. 
            Les algorithmes de ML identifient des patterns dans les données pour faire 
            des prédictions ou prendre des décisions.""",
            
            """Le RAG (Retrieval-Augmented Generation) combine la recherche d'information 
            et la génération de texte. Le système récupère d'abord des documents pertinents 
            puis génère une réponse basée sur ces documents, ce qui améliore la fiabilité."""
        ]
    
    def get_document_stats():
        docs = get_documents()
        return {"count": len(docs), "total_words": 0, "avg_words_per_doc": 0}

# ============================================================
# FONCTIONS DE RETRIEVAL
# ============================================================

@st.cache_resource
def build_index(_embedder, documents):
    """Construit l'index FAISS à partir des documents"""
    # Encoder les documents
    doc_embeddings = _embedder.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # Créer l'index FAISS
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    
    return index, doc_embeddings

def search_documents(query, embedder, index, documents, top_k=3):
    """Recherche les documents les plus pertinents"""
    # Encoder la requête
    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # Recherche dans FAISS
    scores, indices = index.search(query_emb, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "index": int(idx),
                "score": float(score),
                "content": documents[idx]
            })
    
    return results

# ============================================================
# GÉNÉRATION DE RÉPONSE RAG
# ============================================================

def build_rag_prompt(question, retrieved_docs, history=None):
    """Construit le prompt pour le modèle génératif"""
    
    # Documents
    docs_text = ""
    for i, doc in enumerate(retrieved_docs):
        docs_text += f"{doc['content']}\n\n"
    
    # Prompt pour reformulation intelligente
    prompt = f"""Based on the following context, provide a clear and comprehensive answer in French.

Context:
{docs_text}

Question: {question}

Provide a natural, well-formulated answer in French that synthesizes the information:"""
    
    return prompt

def generate_answer(question, retrieved_docs, tokenizer, model, device, history=None, 
                   max_tokens=150, temperature=0.7, num_beams=4):
    """Génère une réponse avec le modèle"""
    
    prompt = build_rag_prompt(question, retrieved_docs, history)
    
    # Tokenisation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Génération avec paramètres optimisés pour réponses longues
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_length=30,  # Force une réponse minimale
            num_beams=num_beams,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.95,
            repetition_penalty=1.1,
            length_penalty=1.5  # Favorise les réponses plus longues
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Si la réponse du modèle est trop courte ou en anglais, utiliser les documents directement
    if len(answer.strip()) < 50 or not any(word in answer.lower() for word in ['le', 'la', 'est', 'les', 'un', 'une']):
        # Construire une réponse à partir des documents
        answer = ""
        for i, doc in enumerate(retrieved_docs[:2]):  # Utiliser les 2 meilleurs documents
            answer += doc['content'] + " "
        answer = answer.strip()
    
    return answer.strip()

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def main():
    st.title("🤖 Chatbot RAG Pédagogique")
    st.markdown("### Assistant intelligent avec système de Retrieval-Augmented Generation")
    
    # Sidebar avec infos et paramètres
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.write("""
        **Modèles utilisés:**
        - Embeddings: paraphrase-multilingual-mpnet-base-v2
        - Génération: flan-t5-base
        
        **Fonctionnalités:**
        - ✅ Recherche sémantique
        - ✅ Index FAISS
        - ✅ Historique de conversation
        - ✅ Réponses basées sur documents
        """)
        
        # Stats sur les documents
        stats = get_document_stats()
        st.divider()
        st.write("**📚 Base documentaire:**")
        st.write(f"- Documents: {stats['count']}")
        st.write(f"- Mots totaux: {stats['total_words']}")
        st.write(f"- Moyenne: {stats['avg_words_per_doc']} mots/doc")
        
        if USE_EXTERNAL_DOCS:
            st.success("✅ Documents chargés depuis documents.py")
        else:
            st.warning("⚠️ Documents par défaut (créer documents.py)")
        
        st.divider()
        
        # ============================================================
        # PARAMÈTRES RÉGLABLES
        # ============================================================
        st.header("⚙️ Paramètres RAG")
        
        st.subheader("Retrieval")
        top_k = st.slider(
            "Nombre de documents à récupérer",
            min_value=1,
            max_value=10,
            value=3,
            help="Nombre de documents les plus similaires à utiliser"
        )
        
        similarity_threshold = st.slider(
            "Seuil de similarité minimum",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Score minimum pour considérer un document (0 = tous)"
        )
        
        st.subheader("Génération")
        max_tokens = st.slider(
            "Longueur maximale de réponse",
            min_value=50,
            max_value=500,
            value=200,  # Augmenté de 150 à 200
            step=10,
            help="Nombre maximum de tokens générés"
        )
        
        temperature = st.slider(
            "Température",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Contrôle la créativité (bas = précis, haut = créatif)"
        )
        
        num_beams = st.slider(
            "Beam Search",
            min_value=1,
            max_value=10,
            value=4,
            help="Nombre de branches pour la génération"
        )
        
        st.divider()
        
        if st.button("🗑️ Réinitialiser la conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()
    
    # Initialisation de la session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Chargement des modèles (avec spinner)
    with st.spinner("Chargement des modèles..."):
        embedder = load_embedding_model()
        tokenizer, gen_model, device = load_generation_model()
        documents = get_documents()
        index, doc_embeddings = build_index(embedder, documents)
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Afficher les documents sources si disponibles
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Documents sources"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"""
                        **Document {i+1}** (score: {doc['score']:.3f})
                        
                        {doc['content'][:200]}...
                        """)
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question..."):
        
        # Afficher le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse..."):
                
                # Retrieval avec paramètres
                retrieved = search_documents(
                    prompt, 
                    embedder, 
                    index, 
                    documents, 
                    top_k=top_k
                )
                
                # Filtrer par seuil de similarité
                retrieved = [doc for doc in retrieved if doc['score'] >= similarity_threshold]
                
                if not retrieved:
                    answer = "❌ Aucun document pertinent trouvé. Essayez de reformuler votre question."
                    st.warning(answer)
                else:
                    # Génération avec paramètres
                    answer = generate_answer(
                        prompt,
                        retrieved,
                        tokenizer,
                        gen_model,
                        device,
                        st.session_state.history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        num_beams=num_beams
                    )
                    
                    # Affichage de la réponse avec style professionnel
                    st.markdown("### Réponse")
                    st.info(answer)
                    
                    # Sources
                    with st.expander(f"📚 Documents sources utilisés ({len(retrieved)} documents)"):
                        for i, doc in enumerate(retrieved):
                            st.markdown(f"""
                            **📄 Document {i+1}** - Score de similarité: `{doc['score']:.3f}`
                            
                            {doc['content']}
                            """)
                            st.divider()
                
                # Mise à jour de l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved if retrieved else []
                })
                st.session_state.history.append((prompt, answer))

if __name__ == "__main__":
    main()