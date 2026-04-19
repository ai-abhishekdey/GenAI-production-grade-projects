import streamlit as st
from src.database import connect, get_summary
from src.chain import build_chain, chat

# Page Config

st.set_page_config(
    page_title="Conversational AI",
    page_icon="🤖",
)

# Initialize (runs once) 

@st.cache_resource
def initialize():
    client = connect()
    chain  = build_chain()
    return client, chain

client, chain = initialize()

# Session State
# Streamlit reruns the whole script on every interaction
# st.session_state persists values across reruns

if "username" not in st.session_state:
    st.session_state.username = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Login Screen

if st.session_state.username is None:
    st.title("🤖 Conversational AI")
    st.subheader("Please enter your username to continue")

    username = st.text_input("Username")

    if st.button("Start Chatting"):
        if username.strip():
            st.session_state.username = username.strip()
            st.rerun()
        else:
            st.error("Username cannot be empty.")

#  Chat Screen

else:
    username = st.session_state.username

    # Header
    st.title("🤖 Conversational AI")
    st.caption(f"Logged in as **{username}**")

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, cb = chat(chain, client, username, user_input)
            st.write(response)
            st.caption(f"Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}")

        # Save AI message to display
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar — summary
    with st.sidebar:
        st.header(f"👤 {username}")

        summary = get_summary(client, username)
        st.metric("Total Messages", summary["messages"])
        st.metric("Total Tokens",   summary["tokens"])
        st.metric("Total Cost",     f"${summary['cost']:.6f}")

        st.divider()

        if st.button("Logout"):
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()
