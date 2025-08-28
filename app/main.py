import streamlit as st
import logging
import os
from utils.logging import setup_logging
from utils.globalVariables import page_icon_path
from utils.llm import TokenUsageHandler

developer_mode = os.getenv("DEVELOPER_MODE", "false").lower() == "true"
if developer_mode:
    setup_logging(logging.DEBUG)
else:
    setup_logging(logging.INFO)

if "token_usage_handler" not in st.session_state:
    st.session_state.token_usage_handler = TokenUsageHandler()

st.set_page_config(page_title="ScribaLLM", page_icon=page_icon_path,
                   layout="centered", initial_sidebar_state="expanded")
settings_page = st.Page('settings.py', title='Settings', icon='⚙️')
ocr_page = st.Page('ocr.py', title='OCR', icon='📝')
chatbot_page = st.Page('chatbot.py', title='Chatbot', icon='💬')
manage_memory_page = st.Page('database.py', title='Manage Memory', icon='📊')

if developer_mode:
    debug_page = st.Page('utils/debug_page.py', title='Debug', icon='👨‍💻')
    pg = st.navigation(
        [ocr_page, chatbot_page, manage_memory_page, settings_page, debug_page])
else:
    pg = st.navigation(
        [ocr_page, chatbot_page, manage_memory_page, settings_page])

pg.run()
