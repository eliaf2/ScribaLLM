import streamlit as st
import logging
from utils.globalVariables import gemini_model_options, openai_model_options

st.markdown('# Settings ⚙️')

# Set up gemini API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

labels = list(gemini_model_options.keys())
gemini_model = st.selectbox(
    '**Select Gemini Model**',
    options=labels,
    index=labels.index(
    next(
        (k for k, v in gemini_model_options.items() if v == st.session_state.get(
            'gemini_llm_model', gemini_model_options[labels[0]])),
        labels[0]
    )),
    help='Choose the Gemini model to use for OCR capabilities.',
)
st.session_state.gemini_llm_model = gemini_model_options[gemini_model]

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

st.markdown('---')
# ==========================

labels=list(openai_model_options.keys())
openai_model=st.selectbox(
    '**Select OpenAI Model**',
    options=labels,
    index=labels.index(
        next(
            (k for k, v in openai_model_options.items() if v == st.session_state.get('openai_llm_model', openai_model_options[labels[0]])),
            labels[0]
        )
    ),
    help='Choose the OpenAI model to use for RAG capabilities.',
)
st.session_state.openai_llm_model = openai_model_options[openai_model]

openai_api_key=st.text_input(
    '**OpenAI API Key**',
    type='password',
    placeholder='api',
    value=st.session_state.get('openai_api_key', ''),
    help='Enter your OpenAI API key to RAG capabilities.',
)

if st.button('Set OpenAI API Key'):
    if openai_api_key:
        st.success('OpenAI API key updated.')
        st.session_state.openai_api_key=openai_api_key
        logging.info('OpenAI API key updated successfully.')
    else:
        st.warning(
            'Please enter your OpenAI API key to enable RAG functionality.')
