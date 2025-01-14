# ************************************************************************************** 
# *********************** IMPORTATION ET CHARGEMENT DES LIBRAIRIES *********************
# **************************************************************************************

import logging, subprocess, nest_asyncio, os, utils, sys, nltk, re, PyPDF2, torch
import llama_index.llms
import random, tempfile
import numpy as np
import pandas as pd
import warnings
import requests

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_index.core import Settings
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from huggingface_hub import login, snapshot_download
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from groq import Groq
from langchain_groq import ChatGroq
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoTokenizer
from llama_parse import LlamaParse
from langchain.prompts import PromptTemplate
from io import BytesIO


warnings.filterwarnings('ignore')

nest_asyncio.apply()
nltk.download('punkt_tab')
# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ************************************************************************************** 
# *************************** INSTANCIATION DES API KEYS *******************************
# **************************************************************************************
login('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')          # Hugging Face API KEY
groq_api_key = "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"     # Groq API KEY



# ************************************************************************************** 
# ***************** CHARGEMENT DE LA BASE DE DONNEES (FICHIERS PDF) ********************
# **************************************************************************************
def download_pdfs_from_github(repo_url, branch, pdf_paths, download_dir):
    """
    Télécharge les fichiers PDF depuis GitHub et les enregistre localement.
    
    :param repo_url: URL de base du dépôt GitHub.
    :param branch: Nom de la branche.
    :param pdf_paths: Liste des chemins des fichiers PDF dans le dépôt.
    :param download_dir: Répertoire local où enregistrer les PDF.
    """
    os.makedirs(download_dir, exist_ok=True)
    base_url = f"{repo_url}/raw/{branch}"

    for pdf_path in pdf_paths:
        file_url = f"{base_url}/{pdf_path}"
        local_path = os.path.join(download_dir, os.path.basename(pdf_path))
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Vérifie si la requête a réussi
            with open(local_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"Fichier téléchargé avec succès : {local_path}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du téléchargement de {file_url}: {e}")


def load_pdfs_from_directory(directory):
    """
    Charge les fichiers PDF depuis un répertoire en utilisant PyPDFDirectoryLoader.
    
    :param directory: Répertoire contenant les fichiers PDF.
    :return: Liste de documents PDF chargés.
    """
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    print(f"{len(documents)} fichiers PDF chargés depuis {directory}")
    return documents


# Paramètres pour les fichiers sur GitHub
repo_url = "https://github.com/kountak/chatbot_douanes"
branch = "main"
pdf_paths = [
    "document_a_fournir_modele.pdf",
    # Ajoute ici les chemins relatifs des autres fichiers PDF
]

# Répertoire temporaire pour télécharger les fichiers PDF
download_dir = "/tmp/pdfs"

# Étape 1 : Télécharger les fichiers PDF depuis GitHub
download_pdfs_from_github(repo_url, branch, pdf_paths, download_dir)

# Étape 2 : Charger les fichiers PDF depuis le répertoire
data = load_pdfs_from_directory(download_dir)

# Chargement des fichiers PDF
# loader = PyPDFDirectoryLoader("https://github.com/kountak/chatbot_douanes/blob/main/")
# data = loader.load()

pdf_texts = [doc.page_content for doc in data]
combined_text = "\n".join(pdf_texts)


# ************************************************************************************** 
# ************************* PARSING ET DECOUPAGE EN CHUNKS *****************************
# **************************************************************************************
instructions = """
            Les documents fournis sont des informations sur les douanes au Togo. Ils contiennent des tableaux.
            Pour les tableaux, les informations contenues sur la première ligne correspond au nom de chaque variable.
            Chaque ligne des tableaux correspond à une information
            Les abréviations, acronymes et leurs définitions présents dans le document doivent être conservés tels quels, car ils sont spécifiques au contexte douanier togolais.
            N'invente pas toi même des définitions aux sigles et acronymes.
            Les codes des régimes douaniers sont spécifiques au Togo et sont composés de 4 chiffres.
            S'il te plait extrait toutes les informations
            N'ajoute aucune information qui ne soit pas contenue dans le document.
            """
parser = LlamaParse(
    api_key="llx-n7EXJpPGOpvO6IFm9oqZbZDxSALcxEd1XeiWLRQchE3L5sTL",
    result_type="markdown",
    verbose=True,
    continuous_mode=True,
    parsing_instruction=instructions,
    max_timeout=10000,
    language = "fr",
)

# Sauvegarder le contenu dans un fichier temporaire au format TXT
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
    temp_file.write(combined_text.encode('utf-8'))  # Écrire en UTF-8
    temp_file_path = temp_file.name

# Charger les données extraites dans un objet Document compatible
from llama_index.core import Document
document = Document(text=combined_text)

# Vérification des informations sur le document
print(f"Type de 'document': {type(document)}\n")
print(f"Taille du texte dans 'document' : {len(document.text)}\n")


# Charger le document dans le parser
parsed_data = parser.load_data(temp_file_path)

# Afficher les données analysées
print("Données analysées par LlamaParse :")
#print(parsed_data)

from langchain.schema import Document
from llama_index.core.schema import MediaResource

# Assurez-vous que les ressources NLTK nécessaires sont téléchargées
nltk.download('punkt')

# Fonction pour diviser le texte en chunks
def chunk_data(parsed_data):
    doc_chunks = []

    # Parcourir chaque entrée dans les données analysées
    for entry in parsed_data:
        logger.info(f"Traitement de l'entrée : {entry.id_}.")

        # Vérifier que text existe et contient un texte
        if hasattr(entry, 'text_resource') and isinstance(entry.text_resource, MediaResource) and hasattr(entry.text_resource, 'text'):
            text = entry.text_resource.text

            logger.info(f"Texte trouvé dans l'entrée {entry.id_}: {text[:50]}...")  # Aperçu du texte

            # Diviser le texte en paragraphes
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                sentences = nltk.sent_tokenize(paragraph, language='french')

                # Traiter chaque phrase
                for sentence in sentences:
                    if sentence.strip():  # Ignore les phrases vides
                        doc_chunks.append(Document(page_content=sentence.strip()))

            logger.info(f"Entrée divisée en {len(doc_chunks)} chunks jusqu'à présent.")
        else:
            logger.warning(f"L'entrée {entry.id_} ne contient pas de texte valide dans 'text_resource'.")

    return doc_chunks

# Exemple d'utilisation avec parsed_data
parsed_data = parser.load_data(temp_file_path)
chunks = chunk_data(parsed_data)
print(f"Nombre total de chunks : {len(chunks)}")

text_chunks = chunk_data(parsed_data)

len(text_chunks)

# ************************************************************************************** 
# ***************** CHARGEMENT DES MODELES D'EMBEDDING ET DU LLM ***********************
# **************************************************************************************
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")  # Embedding model

# LLM model
model_kwargs={"top_p":1}
chat_model = ChatGroq(temperature=0.7,
                      model_name="mixtral-8x7b-32768",
                      model_kwargs=model_kwargs,
                      api_key=groq_api_key,)


# ************************************************************************************** 
# *************************** VECTORISATION DES DONNEES ********************************
# **************************************************************************************
vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
# Utilisez la méthode .index pour accéder à l'index FAISS brut
faiss_index = vectorstore.index

# Afficher les dimensions ou autres propriétés de l'index
print(faiss_index)


# Récupérer tous les vecteurs
stored_vectors = np.array([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)])

# Afficher le nombre de vecteurs stockés
print("Premiers vecteurs stockés :")
print(len(stored_vectors))
 #
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# ************************************************************************************** 
# ********************** DEFINITION PERSONNALISEE DU PROMPT ****************************
# **************************************************************************************

custom_prompt_template = """
    Vous êtes un expert douanier togolais, chargé de répondre à toute les questions sur la douane togolaise.
    Vous devez utiliser uniquement les informations du contexte et de l'expert.
    Utilise les informations du contexte pour répondre aux questions de l'utilisateur.
    Toutes les informations dont tu as besoin pour répondre se trouvent dans le document.
    Il ne faut pas confondre code additionnel, code du régime douanier et code de la taxe.
    Si tu ne connais pas la réponse, réponds "Je ne suis pas en mesure de répondre, veuillez contacter le bureau de douane le plus proche", n'essaie pas d'inventer une réponse.
    Réponds uniquement et uniquement en français.
    Le contexte est récupéré d'une banque d'information et ne fait pas partie de la conversation avec l'utilisateur.
    Les documents fournis sont des informations sur les douanes au Togo. Ils contiennent des tableaux.
    Pour les tableaux, considérez chaque première ligne du tableau comme le nom de la colonne.
    Chaque ligne des tableaux correspond à une information.
    Les abréviations, acronymes et leurs définitions présents dans le document doivent être conservés tels quels, car ils sont spécifiques au contexte douanier togolais.
    N'invente pas toi même des définitions aux sigles et acronymes.
    Les codes des régimes douaniers sont spécifiques au Togo et sont composés de 4 chiffres.
    N'ajoute aucune information qui ne soit pas contenue dans le document.
    Rappelez vous bien le document fourni renferme les informations sur la douane togolaise et que vous etes un expert douanier togolais.

    Context: {context}
    Question: {question}

    Retourne seulement la réponse utile ci-dessous et rien d'autre. Les réponses doivent être uniquement en français.

    Helpful answer:
    """

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Appel de la fonction pour obtenir le prompt
prompt = set_custom_prompt()
print(prompt.template)

# Configuration des paramètres pour la chaîne
chain_type_kwargs = {
    "prompt": prompt  # Utiliser directement le prompt généré
}

# ************************************************************************************** 
# ********************** DEFINITION PERSONNALISEE DU RETRIEVAL**************************
# **************************************************************************************
# Exemple d'initialisation de RetrievalQA avec un modèle de récupération et LLM
chat_model = chat_model
retriever = retriever

def retrieval_qa_chain(llm, prompt, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': prompt
        }
    )

def retrieve_response(query, chat_history=None):
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm=chat_model, prompt=qa_prompt, retriever=retriever)
    
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    logger.info(f"Context utilisé dans le prompt:\n{context}")
    
    llm_response = qa.invoke({"query": query, "chat_history": chat_history or []})
    logger.info(f"Réponse brute du modèle: {llm_response}")
    
    llm_answer = llm_response.get("result", "Pas de réponse trouvée. Veuillez contacter le bureau de douanes le plus proche pour plus d'informations").strip()
    return llm_answer


