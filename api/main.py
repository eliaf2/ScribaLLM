from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes # type: ignore
from model_utils import load_model, predict_layout
import tempfile

app = FastAPI()
model = load_model()


@app.post("/layout-detection/")
async def layout_detection(file: UploadFile = File(...), confidence: float = 0.1):
    file_bytes = await file.read()

    images = convert_from_bytes(file_bytes, dpi=300) # Convert PDF to images

    detections = []
    for img in images:
        det_res = predict_layout(model, img, confidence)
        detections.append(det_res.tojson())

    return JSONResponse(content={"results": detections})
