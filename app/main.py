import streamlit as st
import logging
from utils.logging import setup_logging

setup_logging(logging.DEBUG)

st.set_page_config(page_title="ScribaLLM", page_icon="utils/icon.png")
settings_page = st.Page('settings.py', title='Settings', icon='⚙️')
ocr_page = st.Page('ocr.py', title='OCR', icon='📝')
chatbot_page = st.Page('chatbot.py', title='Chatbot', icon='💬')
debug_page = st.Page('utils/debug_page.py', title='Debug', icon='👨‍💻')

pg = st.navigation([chatbot_page, ocr_page, settings_page, debug_page])

pg.run()
