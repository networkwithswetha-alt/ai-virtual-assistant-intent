"""Streamlit chat UI for the virtual assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st

from assistant import VirtualAssistant

st.set_page_config(page_title="Ava — Virtual Assistant", layout="centered")
st.title("Ava — Virtual Assistant")
st.caption("Intent-classification demo · all training data is synthetic.")

if "bot" not in st.session_state:
    st.session_state.bot = VirtualAssistant()
if "history" not in st.session_state:
    st.session_state.history = [
        ("ava", "Hello! I'm Ava, your virtual assistant. How can I help?")
    ]

for role, msg in st.session_state.history:
    with st.chat_message("assistant" if role == "ava" else "user"):
        st.write(msg)

if prompt := st.chat_input("Type a message..."):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)
    reply = st.session_state.bot.respond(prompt)
    st.session_state.history.append(("ava", reply))
    with st.chat_message("assistant"):
        st.write(reply)

    # show what the classifier saw (transparency for the demo)
    intent, conf = st.session_state.bot.classify(prompt)
    st.caption(f"detected intent: `{intent}` (confidence {conf:.0%})")
