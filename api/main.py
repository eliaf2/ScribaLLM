from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes # type: ignore
from model_utils import load_model, predict_layout
from PIL import Image

app = FastAPI()
model = load_model()


@app.post("/layout-detection/")
async def layout_detection(file: UploadFile = File(...), confidence: float = 0.1) -> JSONResponse:
    '''Detect the layout of a document from a PDF file.

    Parameters
    ----------
    file : UploadFile, optional
        Uploaded file, by default File(...)
    confidence : float, optional
        Confidence threshold for layout detection, by default 0.1

    Returns
    -------
    JSONResponse
        JSON response containing the layout detection results for each page in the PDF.
        It has the following structure:\n
        [
            {\n
                "name": str,\n
                "class": int,\n
                "confidence": float,\n
                "box": {
                    "x1": float,
                    "y1": float,
                    "x2": float,
                    "y2": float
                }\n
            },\n
            ...\n
        ]

        Where:
        - "name": The detected layout element type (e.g., "figure", "plain text")
        - "class": The class index of the detected element (e.g., 1 for "plain text", 3 for "figure")
        - "confidence": The confidence score of the detection
        - "box": The bounding box of the detected element, with:
            - "x1": Top-left x coordinate of the bounding box
            - "y1": Top-left y coordinate of the bounding box
            - "x2": Bottom-right x coordinate of the bounding box
            - "y2": Bottom-right y coordinate of the bounding box
    '''
    file_bytes = await file.read()

    images: list[Image.Image] = convert_from_bytes(file_bytes, dpi=300) # Convert PDF to images

    detections = []
    for img in images:
        det_res = predict_layout(model, img, confidence)
        detections.append(det_res.tojson())

    return JSONResponse(content={"results": detections})
