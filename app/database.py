import streamlit as st
import logging
from utils.globalVariables import my_chroma_config
from utils.llm import ChromaConfig, ChromaVectorStore
import time

st.markdown("# 📊 Manage Memory")

if "memory_vector_store" not in st.session_state:
    st.session_state.memory_vector_store = ChromaVectorStore(my_chroma_config, st.session_state.openai_api_key)

st.write("## Memory Sources")
sources = st.session_state.memory_vector_store.get_source_statistics()
st.dataframe(sources, use_container_width=True)

st.write("### Remove source")
source_to_remove = st.selectbox("Select source to remove", options=sources.index)
if st.button("Remove"):
    source_path = sources.loc[source_to_remove, 'source']
    st.write(source_path)
    st.session_state.memory_vector_store.remove_documents_by_source(source_path)    # type: ignore
    st.success(f"Source '{source_to_remove}' removed successfully.")

    st.session_state.memory_vector_store = ChromaVectorStore(my_chroma_config, st.session_state.openai_api_key)
    st.rerun(scope="fragment")
