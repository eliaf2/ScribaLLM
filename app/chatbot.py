import streamlit as st
import os
from utils.directories import database_data_dir, chroma_filepath
from utils.llm import ChromaConfig, ChatbotLLM
from langchain_core.messages import ToolMessage

st.markdown('# Chatbot 💬')

first_message: str = "How can I help you?"

if st.button("Reset chat"):
    st.session_state["chatbot_messages"] = [
        {"role": "assistant", "content": first_message}]
    st.session_state["user_msgs_position"] = []

if "chatbot_messages" not in st.session_state:
    st.session_state["chatbot_messages"] = [
        {"role": "assistant", "content": first_message}]
if "user_msgs_position" not in st.session_state:
    st.session_state["user_msgs_position"] = []

for msg in st.session_state.chatbot_messages:
    st.chat_message(msg["role"]).markdown(
        msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input():
    chroma_config = ChromaConfig(
        chroma_path=chroma_filepath,
        data_path=database_data_dir,
        chunk_size=1000,
        chunk_overlap=200,
        collection_name="rag_documents",
        file_pattern="*.md",
        embedding_model="text-embedding-3-small",
        batch_size=64,
        force_rebuild=False
    )

    chatbot = ChatbotLLM(
        openai_api_key=st.session_state.openai_api_key, chroma_config=chroma_config)
    st.session_state.chatbot_messages.append(
        {"role": "user", "content": prompt})
    st.session_state.user_msgs_position.append(
        len(st.session_state.chatbot_messages)-1)
    st.chat_message("user").write(prompt)

    response = chatbot(st.session_state.chatbot_messages,
                       st.session_state.user_msgs_position)
    response_last_message = response["messages"][-1].content

    contains_tool_message = any(isinstance(msg, ToolMessage)
                                for msg in response["messages"])

    if contains_tool_message:
        st.chat_message("assistant", avatar="utils/icon.png").markdown(
            response_last_message, unsafe_allow_html=True)
    else:
        st.chat_message("assistant").markdown(
            response_last_message, unsafe_allow_html=True)
