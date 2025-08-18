import streamlit as st
import requests
import json
import logging
import base64
import os
import re
import io
import zipfile
import matplotlib.pyplot as plt
from pdf2image import convert_from_bytes  # type: ignore
from matplotlib.patches import Rectangle
from streamlit.runtime.uploaded_file_manager import UploadedFile
from utils.globalVariables import *
from utils.llm import OCR_LLM, ImageOut, ChromaVectorStore
from PIL import Image


def get_metadata(uploaded_file: UploadedFile, save: bool = False) -> dict:
    '''Get metadata of the uploaded file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded file object.
    save : bool, optional
        Whether to save the metadata to a file, by default False

    Returns
    -------
    dict
        The metadata of the uploaded file.
    '''
    global tmp_dir
    metadata = {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "file_id": uploaded_file.file_id
    }
    if save:
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, f"metadata.json"), "w") as f:
            json.dump(metadata, f)
    return metadata


def load_metadata() -> dict | None:
    '''Load metadata from the metadata.json file.

    Returns
    -------
    dict | None
        The metadata of the uploaded file or None if the file does not exist.
    '''
    global tmp_dir
    try:
        with open(os.path.join(tmp_dir, f"metadata.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_resource
def call_layout_detection_api(file: bytes, confidence: float) -> dict | None:
    '''Call the layout detection API.

    Parameters
    ----------
    file : bytes
        Image in bytes format from pdf file to analyzed.
    confidence : float
        Confidence threshold for layout detection.

    Returns
    -------
    dict | None
        JSON response from the API containing layout detection results or None if an error occurs.

    Example of output:
    {'results': [{'name': 'isolate_formula', 'class': 8, 'confidence': 0.50954, 'box': {'x1': 149.06, 'y1': 201.92, 'x2': 1560.45, 'y2': 328.20}}, ...]}
    '''
    logging.debug("Calling layout detection API")
    response = requests.post(
        "http://api:8000/layout-detection/",
        files={"file": file},
        params={"confidence": confidence}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error calling API")
        return None


def filter_detection(detections: list, types: list[str] = []) -> list:
    '''Filter the detection results based on the specified types.

    Parameters
    ----------
    detections : list
        List of detection results.
    types : list[str], optional
        List of types to filter the detections. If empty, all detections are returned.

    Returns
    -------
    list
        List of filtered detection results.
    '''
    if not types:
        return [json.loads(det) if isinstance(det, str) else det for det in detections]
    filter_detection = []
    for page in range(len(detections)):
        page_filter_detection = []
        page_detection = json.loads(detections[page]) if isinstance(
            detections[page], str) else detections[page]
        for det in page_detection:
            if det["name"] in types:
                page_filter_detection.append(det)

        filter_detection.append(page_filter_detection)

    return filter_detection if filter_detection else []


# @st.cache_resource
def ocr_llm() -> tuple[list[str] | None, OCR_LLM]:
    '''Perform OCR on cropped images using llm models. Look at the OCR_LLM class in utils/llm.py for more details.

    Returns
    -------
    tuple[list[str] | None, OCR_LLM]
        A tuple containing a list of OCR results in markdown format or None if no text is detected, and the OCR_LLM instance used for processing.
    '''
    openai_api_key = st.session_state.openai_api_key
    openai_llm_model = st.session_state.openai_llm_model
    gemini_api_key = st.session_state.gemini_api_key
    gemini_llm_model = st.session_state.gemini_llm_model

    ocr = OCR_LLM(openai_api_key=openai_api_key,
                  openai_llm_model=openai_llm_model,
                  gemini_api_key=gemini_api_key, 
                  gemini_llm_model=gemini_llm_model,
                  context=st.session_state.context)
    ocr_results = list()
    for page_folder in sorted(os.listdir(tmp_crop_dir)):
        cropped_images_path = os.path.join(tmp_crop_dir, page_folder)
        if os.path.isdir(cropped_images_path):
            logging.debug(f"Performing OCR on image: {cropped_images_path}")
            try:
                result = ocr(os.path.join(
                    tmp_jpg_dir, f"{page_folder}.jpg"), cropped_images_path)
                if result:
                    ocr_results.append(result)
            except Exception as e:
                logging.error(
                    f"Error performing OCR on {cropped_images_path}: {e}")
                st.error(
                    f"Error performing OCR on {cropped_images_path}. Check logs for details.")

    if ocr_results:
        # return "\n".join(ocr_results)
        return ocr_results, ocr
    else:
        logging.warning("No OCR results found.")
        st.warning(
            "No text detected in the images. Please check the cropped images.")
        return None, ocr


def plot_detection_frames(page: Image.Image, det_json: list) -> None:
    '''Plot the detection frames on the given page image.

    Parameters
    ----------
    page : Image.Image
        Image of the page where detections will be plotted.
    det_json : list
        List containing the detection results with bounding boxes and labels. Each element must be in the format:
        {
            "name": str,
            "class": int,
            "box": {
                "x1": int,
                "y1": int,
                "x2": int,
                "y2": int
            },

            "confidence": float
        }\n
        for each detection.\nThe ```class``` field is not used in this function.
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(page)
    ax.axis('off')

    for det in det_json:
        box = det["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='C0',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 5,
            f'{det["name"]} ({det["confidence"]:.2f})',
            color='C0',
            fontsize=8,
            backgroundcolor='white'
        )

    st.pyplot(fig)


def clear_directory(directory_path: str) -> None:
    '''Removes all files in the ```directory_path```.

    Parameters
    ----------
    directory_path : str
        Path to the directory to be cleaned.
    '''
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_directory(file_path)
            os.rmdir(file_path)
    logging.debug(f"Cleaned directory: {directory_path}")


def save_cropped_images():
    '''
    Save cropped images from the ```tmp_jpg_dir``` directory to the ```tmp_crop_dir``` directory based on the detection results.
    '''
    global tmp_jpg_dir, tmp_crop_dir
    os.makedirs(tmp_crop_dir, exist_ok=True)
    clear_directory(tmp_crop_dir)
    logging.debug(f"Saving cropped images in {tmp_crop_dir}")

    for filename in os.listdir(tmp_jpg_dir):
        if filename.lower().endswith(".jpg"):
            jpg_path = os.path.join(tmp_jpg_dir, filename)
            page_index = int(filename.split('.')[0])
            det_json = st.session_state.detection_results[page_index]
            img = Image.open(jpg_path)

            page_crop_dir = os.path.join(tmp_crop_dir, str(page_index))
            os.makedirs(page_crop_dir, exist_ok=True)

            counter = 0
            det_list = sorted(det_json, key=lambda det: det["box"]["y1"])
            for det in det_list:
                box = det["box"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_path = os.path.join(
                    page_crop_dir, f"{counter}.jpg")
                cropped_img.save(cropped_path)
                logging.debug(f"Cropped image saved: {cropped_path}")
                counter += 1


def plot_cropped_images():
    '''Display cropped images from the ```tmp_crop_dir``` directory in the Streamlit app.
    '''
    global tmp_crop_dir
    if not os.listdir(tmp_crop_dir):
        st.warning("No cropped images found. Please perform OCR first.")
        return

    for page_num in sorted(os.listdir(tmp_crop_dir)):
        page_folder_path = os.path.join(tmp_crop_dir, page_num)
        for filename in os.listdir(page_folder_path):
            if filename.lower().endswith(".jpg"):
                img_path = os.path.join(page_folder_path, filename)
                st.image(img_path, caption=filename, use_container_width=False)
                logging.debug(f"Displaying cropped image: {img_path}")


def clear_md(text: str) -> str:
    '''Clear Markdown formatting from the text.

    Parameters
    ----------
    text : str
        The input text with Markdown formatting.

    Returns
    -------
    str
        The text without Markdown formatting.
    '''
    if text.startswith("```markdown"):
        text = text[11:]  # Remove "```markdown" at beginning of the string
    if text.endswith("```"):
        text = text[:-3]  # Remove "```" at the end of the string
    return text


def get_base64_image(image_path: str) -> str | None:
    '''Converts a local image to base64

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    str | None
        Base64-encoded image string or None if the image is not found.
    '''
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None


def fix_url_pictures(markdown_text: str) -> str:
    '''Fixes corrupted URLs in the Markdown text.

    Parameters
    ----------
    markdown_text : str
        The input Markdown text containing image references.

    Returns
    -------
    str
        The Markdown text with fixed URLs.
    '''
    return re.sub(r'(?<!/)(ScribaLLM/tmp/cropped)', r'/\1', markdown_text)


def convert_markdown_images_to_base64(markdown_text: str, clear: bool = False) -> str:
    '''Converts Markdown image references to base64 HTML

    Parameters
    ----------
    markdown_text : str
        The input Markdown text containing image references.
    clear : bool, optional
        Whether to clear Markdown formatting from the text, by default False

    Returns
    -------
    str
        The HTML output with images converted to base64.
    '''
    def replace_image(match: re.Match) -> str:
        alt_text: str = match.group(1)
        image_path: str = match.group(2)

        if os.path.exists(image_path):
            img_base64 = get_base64_image(image_path)
            if img_base64:
                ext: str = os.path.splitext(image_path)[1].lower()
                mime_type: str = "image/jpeg" if ext in [
                    '.jpg', '.jpeg'] else f"image/{ext[1:]}"

                return f'<img src="data:{mime_type};base64,{img_base64}" alt="{alt_text}" style="max-width: 70%; height: auto;">'

        # If the image does not exist, show a placeholder
        logging.warning(
            f"Markdown line with missing image: ![{alt_text}]({image_path})")
        return f'<div style="border: 2px dashed #ccc; padding: 20px; text-align: center;">⚠️ Markdown line with missing image: <code>![{alt_text}]({image_path})</code></div>'

    if clear:
        markdown_text = clear_md(markdown_text)

    # Pattern to find ![alt text](path/to/image.ext)
    pattern: str = r'!\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, replace_image, markdown_text)


def make_zip_file(text: str, list_images: list[ImageOut]) -> bytes:
    '''Creates a ZIP file containing the text and images.

    Parameters
    ----------
    text : str
        The text content to include in the ZIP file.
    list_images : list[ImageOut]
        The list of images to include in the ZIP file.

    Returns
    -------
    bytes
        The bytes of the created ZIP file.
    '''
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for img in list_images:
            with open(img.path, "rb") as img_file:
                image_path_zip = os.path.join(
                    "img", os.path.relpath(img.path, tmp_crop_dir))
                text = text.replace(img.path, image_path_zip)
                zip_file.writestr(image_path_zip, img_file.read())

        zip_file.writestr("text.md", text.encode('utf-8'))

    return zip_buffer.getvalue()


def add_results_to_database(results: str) -> None:
    '''Adds the OCR results to the database and update embeddings database.

    Parameters
    ----------
    results : str
        The OCR results to add to the database.
    '''
    global database_data_dir, tmp_dir

    if not os.path.exists(database_data_dir):
        os.makedirs(database_data_dir)

    with open(os.path.join(tmp_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    filename = os.path.splitext(metadata['name'])[0]
    file_path = os.path.join(database_data_dir, f"{filename}.md")

    with open(file_path, "w") as f:
        f.write(results)

    vector_store = ChromaVectorStore(
        my_chroma_config, st.session_state.openai_api_key)
    vector_store.generate_data_store()

    stats = vector_store.get_database_stats()
    stats_log = "\n".join(
        f"{key.replace('_', ' ').title()}: {value}" for key, value in stats.items())
    logging.info(f"Database Stats:\n{stats_log}")
    st.success(f"Results added to database: {file_path}\nMemory updated.")


# ================ Page display ================
st.write('# OCR 📝')

st.write('## Upload a PDF file to perform OCR')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
uploaded_file_name = uploaded_file.name if uploaded_file else None

if 'page_index' not in st.session_state:
    st.session_state.page_index = 0
if 'uploaded_file_metadata' not in st.session_state:
    st.session_state.uploaded_file_metadata = {}

if uploaded_file is not None and ('uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_file_name):
    st.session_state.uploaded_file_metadata = get_metadata(
        uploaded_file, save=True)
    st.session_state["uploaded_filename"] = uploaded_file.name
    st.session_state.page_index = 0

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_jpg_dir, exist_ok=True)
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    pages = convert_from_bytes(uploaded_file.getvalue(), dpi=300)

    clear_directory(tmp_jpg_dir)

    for i, page in enumerate(pages):
        jpg_path = os.path.join(tmp_jpg_dir, f"{i}.jpg")
        page.save(jpg_path, "JPEG")

    st.success(f"File uploaded. Sending to backend...")
else:
    st.session_state.uploaded_file_metadata = load_metadata()

st.write('## Layout Detection')

if "conf" not in st.session_state:  # Set the confidence threshold
    st.session_state.conf = 0.1

st.session_state.conf = st.slider(
    'Confidence Threshold',
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.conf,
    step=0.01,
    help='Adjust the confidence threshold for layout detection.'
)

with open(tmp_file_path, "rb") as file:
    tmp_file: bytes = file.read()
response = call_layout_detection_api(tmp_file, st.session_state.conf)
response = filter_detection(response["results"], types=[
                            'figure']) if response else []

if response:
    st.session_state.detection_results = response
    st.session_state.pages = convert_from_bytes(tmp_file, dpi=300)
else:
    st.error("No results found or error in response.")
    st.session_state.detection_results = []
    st.session_state.pages = []

if "detection_results" in st.session_state:
    num_pages = len(st.session_state.detection_results)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Prev", disabled=st.session_state.page_index == 0):
            st.session_state.page_index = max(
                0, st.session_state.page_index - 1)

    with col3:
        if st.button("Next ➡️", disabled=st.session_state.page_index == num_pages - 1):
            st.session_state.page_index = min(
                num_pages - 1, st.session_state.page_index + 1)

    with col2:
        selected_page = st.text_input(
            "Page Number",
            value=str(st.session_state.page_index + 1),
            help=f"Enter a page number (1 to {num_pages}) to view its layout detection."
        )
        if selected_page.isdigit():
            st.session_state.page_index = max(
                0, min(num_pages - 1, int(selected_page) - 1))

    st.write(f"Page {st.session_state.page_index + 1} of {num_pages}")

    page = st.session_state.pages[st.session_state.page_index]
    det_json = st.session_state.detection_results[st.session_state.page_index]

    save_cropped_images()
    plot_detection_frames(page, det_json)   # type: ignore

    st.write("### Layout Detection Results")
    plot_cropped_images()

st.write('# Conversion')
if "context" in st.session_state:
    context = st.text_input("Context", value=st.session_state.context)
else:
    context = st.text_input(
        "Context", placeholder="Write a brief summary of the PDF content here.")
    if context:
        st.session_state.context = context

if "ocr_output" not in st.session_state:
    st.session_state.ocr_output = []

if st.button("Convert to Text"):
    save_cropped_images()
    st.session_state.ocr_output, llm = ocr_llm()

    st.session_state.ocr_output_text_list = []
    st.session_state.ocr_output_pictures_list = []
    for page in st.session_state.ocr_output:    # type: ignore
        st.session_state.ocr_output_text_list.append(page[0])
        st.session_state.ocr_output_pictures_list.append(page[1])

    st.session_state.ocr_output_pictures_list = [
        item for sublist in st.session_state.ocr_output_pictures_list for item in sublist
    ]

    st.session_state.ocr_output = llm.improve_ocr_result(
        st.session_state.ocr_output_text_list, st.session_state.context)
    st.session_state.ocr_output = fix_url_pictures(st.session_state.ocr_output)
    logging.debug("OCR conversion completed.")
    logging.debug(f"OCR results: {st.session_state.ocr_output}")

st.write("### OCR Results")
if st.session_state.ocr_output == []:
    st.markdown("**Click the button to convert the PDF to text using OCR.**")
else:
    if isinstance(st.session_state.ocr_output, str):
        st.markdown(convert_markdown_images_to_base64(
            st.session_state.ocr_output, clear=False), unsafe_allow_html=True)

        st.download_button(
            label="📥 Download",
            data=make_zip_file(st.session_state.ocr_output,
                               st.session_state.ocr_output_pictures_list),
            file_name="output.zip",
            mime="application/zip",
            help="Click to download the OCR results as a ZIP file."
        )

        _filename_md = os.path.splitext(st.session_state.uploaded_file_metadata['name'])[0]     # type: ignore
        _file_path = os.path.join(database_data_dir, f"{_filename_md}.md")

        if os.path.exists(_file_path):
            if st.button(
                label="📝 Overwrite memory",
                help="Click to overwrite the chatbot memory about this file."
            ):
                add_results_to_database(st.session_state.ocr_output)
        else:
            if st.button(
                label="🧠 Add to memory",
                help="Click to add this file to the chatbot memory."
            ):
                add_results_to_database(st.session_state.ocr_output)
