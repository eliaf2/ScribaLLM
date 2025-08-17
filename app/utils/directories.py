import os

tmp_dir: str = "/ScribaLLM/tmp"
database_dir: str = "/ScribaLLM/database"

tmp_file_path: str = os.path.join(tmp_dir, "uploaded_file.pdf")
tmp_jpg_dir: str = os.path.join(tmp_dir, "jpg")
tmp_crop_dir: str = os.path.join(tmp_dir, "cropped")
database_data_dir: str = os.path.join(database_dir, "data")
chroma_filepath: str = os.path.join(database_dir, "chroma")