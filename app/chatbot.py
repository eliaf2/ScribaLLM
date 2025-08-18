import streamlit as st
import logging
import os
from utils.globalVariables import my_chroma_config
from utils.llm import ChromaConfig, ChatbotLLM
from langchain_core.messages import ToolMessage

st.markdown('# Chatbot 💬')

if "openai_api_key" not in st.session_state:
    st.error("Please provide your OpenAI API key to use the chatbot.")
else:
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
        st.chat_message(msg["role"], avatar=msg.get("avatar")).markdown(
            msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input():

        chatbot = ChatbotLLM(
            openai_api_key=st.session_state.openai_api_key, openai_llm_model=st.session_state.openai_llm_model, chroma_config=my_chroma_config)
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
            st.session_state.chatbot_messages.append(
                {"role": "assistant", "avatar": "utils/icon.png", "content": response_last_message})
        else:
            st.chat_message("assistant").markdown(
                response_last_message, unsafe_allow_html=True)
            st.session_state.chatbot_messages.append(
                {"role": "assistant", "content": response_last_message})
