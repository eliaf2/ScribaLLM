import streamlit as st
import io

st.set_page_config(page_title="Debug Page", layout="wide")

st.title("Debug Page")

ocr_log = "/ScribaLLM/logs/main.log"
try:
    with open(ocr_log, "r") as log_file:
        log_content = log_file.readlines()

    st.subheader("Log File Content")
    log_output = "".join(log_content)

    # Display log content in a text area and scroll to the bottom by default - Must be FIXED
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
        with open(ocr_log, "w") as log_file:
            log_file.write("")
        st.success("Log file cleaned successfully!")


    st.download_button(
            label="Download Log File",
            data=log_output,
            file_name="main.log",
            mime="text/plain")


except FileNotFoundError:
    st.warning(f"Log file not found at {ocr_log}")
except Exception as e:
    st.error(f"Error reading log file: {e}")