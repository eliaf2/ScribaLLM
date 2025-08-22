import streamlit as st
import io
import pandas as pd
from utils.globalVariables import log_path

st.set_page_config(page_title="Debug Page", layout="wide")

st.title("Debug Page")

try:
    with open(log_path, "r") as log_file:
        log_content = log_file.readlines()

    st.subheader("Log File Content")
    log_output = "".join(log_content)

    # Display log content in a text area and scroll to the bottom by default
    st.text_area("Log Output", log_output, height=400, key="log_output")
    st.markdown(
        """
        <script>
        const textarea = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
        if (textarea) {
            textarea.scrollTop = textarea.scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Clean Log File"):
        with open(log_path, "w") as log_file:
            log_file.write("")
        st.success("Log file cleaned successfully.")


    st.download_button(
            label="Download Log File",
            data=log_output,
            file_name="main.log",
            mime="text/plain")


except FileNotFoundError:
    st.warning(f"Log file not found at {log_path}")
except Exception as e:
    st.error(f"Error reading log file: {e}")

st.write("## Session State Keys")
if st.session_state:
    session_state_df = pd.DataFrame(
        [
            {
                "Key": key,
                "Value": (str(value)[:100] + "...") if len(str(value)) > 100 else str(value),
            }
            for key, value in st.session_state.items()
        ]
    )
    st.table(session_state_df)
else:
    st.write("No session state keys available.")
