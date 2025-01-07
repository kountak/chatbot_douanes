import streamlit as st
from typing import List
import base64
import time

from main import retrieve_response

# Fonction pour définir l'image de fond d'écran
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
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
set_background_image('/home/onyxia/work/fond.jpg')

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
        time.sleep(0.1)

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

    # Réponse automatique pour les salutations
    salutations = ["Bonjour", "Coucou", "Bjr", "Cc", "Hello", "Hi"]
    if prompt in salutations:
        assistant_response = "Bonjour, je suis **Sinayo**, votre agent conversationnel spécialisé sur les questions douanières au Togo, en quoi puis-je vous aider ?"
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.markdown(
            f"""
            <div class="message-container assistant-container">
                <div class="message assistant-message">{assistant_response}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        try:
            chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
            response = retrieve_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(
                f"""
                <div class="message-container assistant-container">
                    <div class="message assistant-message">{response}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            error_message = f"Désolé, une erreur s'est produite : {str(e)}. Veuillez réessayer plus tard."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.markdown(
                f"""
                <div class="message-container assistant-container">
                    <div class="message assistant-message">{error_message}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
