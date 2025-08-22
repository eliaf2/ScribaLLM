import os
from utils.llm import ChromaConfig

log_dir = "/ScribaLLM/logs"
log_path = "/ScribaLLM/logs/main.log"

tmp_dir: str = "/ScribaLLM/tmp"
database_dir: str = "/ScribaLLM/database"

tmp_file_path: str = os.path.join(tmp_dir, "uploaded_file.pdf")
tmp_jpg_dir: str = os.path.join(tmp_dir, "jpg")
tmp_crop_dir: str = os.path.join(tmp_dir, "cropped")
database_data_dir: str = os.path.join(database_dir, "data")
chroma_filepath: str = os.path.join(database_dir, "chroma")

my_chroma_config: ChromaConfig = ChromaConfig(
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

gemini_model_options: dict = {
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.5 Flash": "gemini-2.5-flash"
}

openai_model_options: dict = {
    "GPT 4.1 Nano": "gpt-4.1-nano",
    "GPT 4.1 Mini": "gpt-4.1-mini"
}

avatar_icon_path: str = "utils/icon_page.svg"
page_icon_path: str = "utils/icon_page.svg"