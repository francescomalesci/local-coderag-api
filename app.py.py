import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

# Load configuration
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Page configuration
st.set_page_config(page_title="Local CodeRAG", page_icon="🤖")
st.title("Local CodeRAG")

mode = st.radio(
    "Processing Architecture:",
    ("Simple RAG (Fast, database only)", "Intelligent Agent (Slow, uses external tools)"),
    horizontal=True
)

# Set URL based on user choice
if "Simple RAG" in mode:
    API_URL = "http://127.0.0.1:8000/chat"
else:
    API_URL = "http://127.0.0.1:8000/agent"

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message:
            st.caption(message["meta"])

# User input
if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        # Cambiato 'domanda' in 'question'
        payload = {"question": prompt, "session_id": "demo_session_1"}
        meta = {}

        def token_generator():
            try:
                with requests.post(API_URL, json=payload, headers=headers, stream=True) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event["type"] == "token":
                            yield event["content"]
                        elif event["type"] == "done":
                            meta.update(event)
                        elif event["type"] == "error":
                            yield f"\n\n⚠️ Error: {event['content']}"
            except Exception as e:
                yield f"\n\n⚠️ API connection error: {e}"

        full_response = st.write_stream(token_generator())

        if meta:
            # Cambiate le chiavi estratte da meta.get()
            meta_info = f"Source: {meta.get('source_used', '?')} | Time: {meta.get('execution_time_seconds', '?')}s"
            st.caption(meta_info)
            st.session_state.messages.append({"role": "assistant", "content": full_response, "meta": meta_info})
	else:
            st.session_state.messages.append({"role": "assistant", "content": full_response})