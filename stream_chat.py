import streamlit as st
import requests
from PIL import Image
from typing import List
import base64
import time

from main import retrieve_response

# Fonction pour définir l'image de fond d'écran
def set_background_image(image_url):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        # Encoder l'image en base64
        encoded_string = base64.b64encode(response.content).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{ 
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True 
        )
    else:
        st.error("Impossible de charger l'image. Vérifiez l'URL.")
    st.markdown(
        f"""
        <style>        
        .message-container {{
            display: flex;
            flex-direction: row;
            margin-bottom: 10px;
        }}
        .user-container {{
            justify-content: flex-end;
        }}
        .assistant-container {{
            justify-content: flex-start;
        }}
        .message {{
            max-width: 70%;
            padding: 10px;
            border-radius: 10px;
            word-wrap: break-word;
        }}
        .user-message {{
            background-color: #dcf8c6; /* Couleur des messages utilisateur (WhatsApp style) */
            align-self: flex-end;
        }}
        .assistant-message {{
            background-color: #f1f0f0; /* Couleur des messages assistant */
            align-self: flex-start;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Titre de l'application
st.title("Togo - ChatBot Douanes")
st.markdown(
    "*Bonjour, je suis **Sinayo**, votre agent conversationnel spécialisé sur les questions douanières au Togo, en quoi puis-je vous aider ?*", 
    unsafe_allow_html=True
)

# Définir l'image de fond d'écran (remplacez 'background.jpg' par le chemin de votre image)
set_background_image('https://raw.githubusercontent.com/kountak/chatbot_douanes/main/fond.png')

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages de l'historique
for message in st.session_state.messages:
    role_class = "user-container" if message["role"] == "user" else "assistant-container"
    message_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(
        f"""
        <div class="message-container {role_class}">
            <div class="message {message_class}">{message["content"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Fonction pour afficher la réponse mot par mot avec des puces (...)
def display_response(response):
    if "\n" in response:  # Divise en plusieurs points si des sauts de ligne sont détectés
        points = response.split("\n")
        response = " (...) ".join(points)
    words = response.split()
    displayed_response = ""
    response_container = st.empty()
    for word in words:
        displayed_response += word + " "
        response_container.markdown(displayed_response, unsafe_allow_html=True)
        time.sleep(0.05)

# Réaction à une nouvelle question
if prompt := st.chat_input("Posez votre question ici"):
    # Affichage de la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"""
        <div class="message-container user-container">
            <div class="message user-message">{prompt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gestion des salutations
    salutations = ["Bonjour", "Coucou", "Bjr", "Cc", "Hello", "Hi"]
    if prompt in salutations:
        response = "Bonjour, je suis **Sinayo**, votre agent conversationnel spécialisé sur les questions douanières au Togo, en quoi puis-je vous aider ?"
    else:
        try:
            # Appeler la fonction de récupération de réponse
            response = retrieve_response(prompt)
        except Exception as e:
            # Initialisation de la réponse en cas d'erreur
            response = f"Désolé, une erreur s'est produite : {str(e)}. Veuillez réessayer plus tard."

    # Ajouter la réponse dans l'historique et l'afficher
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(
        f"""
        <div class="message-container assistant-container">
            <div class="message assistant-message">{response}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
