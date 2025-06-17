import streamlit as st
import logging

st.markdown('# Settings ⚙️')

# Set up gemini API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

gemini_api_key = st.text_input(
    '**Gemini API Key**',
    type='password',
    placeholder='api',
    value=st.session_state.get('gemini_api_key', ''),
    help='Enter your Gemini API key to use OCR capabilities.',
)

if st.button('Set Gemini API Key'):
    if gemini_api_key:
        st.success('Gemini API key updated.')
        st.session_state.gemini_api_key = gemini_api_key
        logging.info('Gemini API key updated successfully.')
    else:
        st.warning(
            'Please enter your Gemini API key to enable OCR functionality.')

# Set up OpenAI API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

openai_api_key = st.text_input(
    '**OpenAI API Key**',
    type='password',
    placeholder='api',
    value=st.session_state.openai_api_key,
    help='Enter your OpenAI API key to RAG capabilities.',
)

if st.button('Set OpenAI API Key'):
    if openai_api_key:
        st.success('OpenAI API key updated.')
        st.session_state.openai_api_key = openai_api_key
        logging.info('OpenAI API key updated successfully.')
    else:
        st.warning(
            'Please enter your OpenAI API key to enable RAG functionality.')
