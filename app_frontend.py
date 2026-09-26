import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Caricamento configurazione
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = "http://127.0.0.1:8000/chat"

# Configurazione pagina
st.set_page_config(page_title="Local CodeRAG", page_icon="🤖")
st.title("Local CodeRAG")

# Inizializzazione cronologia chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Rendering messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message:
            st.caption(message["meta"])

# Input utente
if prompt := st.chat_input("Fai una domanda sui documenti..."):
    # Mostra domanda utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chiamata all'API backend
    with st.chat_message("assistant"):
        with st.spinner("Elaborazione in corso..."):
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            payload = {"domanda": prompt}
            
            try:
                response = requests.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                dati = response.json()
                
                risposta = dati["risposta"]
                meta_info = f"Fonte: {dati['fonte_utilizzata']} | Tempo: {dati['tempo_esecuzione_secondi']}s"
                
                st.markdown(risposta)
                st.caption(meta_info)
                
                # Salvataggio risposta
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": risposta,
                    "meta": meta_info
                })
            except Exception as e:
                st.error(f"Errore di connessione all'API: {e}")