import streamlit as st
import requests
import json
import logging
import matplotlib.pyplot as plt
from pdf2image import convert_from_bytes  # type: ignore
import os
from matplotlib.patches import Rectangle
from utils.llm import OCR_LLM
from PIL import Image

tmp_dir: str = "/ScribaLLM/tmp"
tmp_file_path: str = os.path.join(tmp_dir, "uploaded_file.pdf")
tmp_jpg_dir: str = os.path.join(tmp_dir, "jpg")
tmp_crop_dir = os.path.join(tmp_dir, "cropped")

@st.cache_resource
def call_layout_detection_api(file: bytes, confidence: float) -> dict | None:
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


# @st.cache_resource
def ocr_llm():
    openai_api_key = st.session_state.openai_api_key
    google_api_key = st.session_state.gemini_api_key

    ocr = OCR_LLM(openai_api_key=openai_api_key, google_api_key=google_api_key, context="")
    ocr_results = list()
    for filename in os.listdir(tmp_crop_dir):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(tmp_crop_dir, filename)
            logging.debug(f"Performing OCR on image: {image_path}")
            try:
                result = ocr(image_path)
                if result:
                    ocr_results.append(result)
            except Exception as e:
                logging.error(f"Error performing OCR on {image_path}: {e}")
                st.error(f"Error performing OCR on {image_path}. Check logs for details.")
    
    if ocr_results:
        # return "\n".join(ocr_results)
        return ocr_results, ocr
    else:
        logging.warning("No OCR results found.")
        st.warning("No text detected in the images. Please check the cropped images.")
        return None, ocr


def plot_detection_frames(page, det_json):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(page)
    ax.axis('off')

    for det in json.loads(det_json):
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

def clean_directory(directory_path: str):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    logging.debug(f"Cleaned directory: {directory_path}")

def save_cropped_images():
    global tmp_jpg_dir, tmp_crop_dir
    os.makedirs(tmp_crop_dir, exist_ok=True)
    clean_directory(tmp_crop_dir)
    logging.debug(f"Saving cropped images in {tmp_crop_dir}")
    
    for filename in os.listdir(tmp_jpg_dir):
        if filename.lower().endswith(".jpg"):
            jpg_path = os.path.join(tmp_jpg_dir, filename)
            page_index = int(filename.split('.')[0])
            det_json = st.session_state.detection_results[page_index]
            img = Image.open(jpg_path)

            counter = 0
            det_list = sorted(json.loads(det_json), key=lambda det: det["box"]["y1"])
            for det in det_list:
                box = det["box"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_path = os.path.join(tmp_crop_dir, f"{page_index}_{counter}.jpg")  # First number is the page index, second is the index of the element
                cropped_img.save(cropped_path)
                logging.debug(f"Cropped image saved: {cropped_path}")
                counter += 1

def plot_cropped_images():
    global tmp_crop_dir
    st.write("### Cropped Images")
    if not os.listdir(tmp_crop_dir):
        st.warning("No cropped images found. Please perform OCR first.")
        return

    for filename in sorted(os.listdir(tmp_crop_dir)):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(tmp_crop_dir, filename)
            st.image(img_path, caption=filename, use_container_width=False)
            logging.debug(f"Displaying cropped image: {img_path}")

# Page display
st.write('# OCR 📝')

st.write('## Upload a PDF file to perform OCR')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
uploaded_file_name = uploaded_file.name if uploaded_file else None

if 'page_index' not in st.session_state:
    st.session_state.page_index = 0  


if uploaded_file is not None and ('uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != uploaded_file_name):
    st.session_state["uploaded_filename"] = uploaded_file.name
    st.session_state.page_index = 0 

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_jpg_dir, exist_ok=True)
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue()) 

    pages = convert_from_bytes(uploaded_file.getvalue(), dpi=300) 

    clean_directory(tmp_jpg_dir)

    for i, page in enumerate(pages):
        jpg_path = os.path.join(tmp_jpg_dir, f"{i}.jpg")
        page.save(jpg_path, "JPEG")

    st.success(f"File uploaded. Sending to backend...")

st.write('## Layout Detection')

if "conf" not in st.session_state:
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

if response and "results" in response:
    st.session_state.detection_results = response["results"]
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
    plot_detection_frames(page, det_json)

    st.write("### Layout Detection Results")
    plot_cropped_images()

st.write('# Conversion')

if st.button("Convert to Text"):
    save_cropped_images()
    ocr_list, ocr_model = ocr_llm()
    logging.debug("OCR conversion completed.")
    logging.debug(f"OCR results: {ocr_list}")

    refined_text = ocr_model.refine(ocr_list) if ocr_list else None

    if ocr_list:
        text_ocr = "<br>".join(map(str, ocr_list))
    else:
        text_ocr = None

    if text_ocr:
        st.write("### OCR Result")
        st.markdown(refined_text, unsafe_allow_html=True)
    else:
        st.error("OCR failed or no text detected.")
else:
    st.write("Click the button to convert the PDF to text using OCR.")
