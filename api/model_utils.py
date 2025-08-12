from doclayout_yolo import YOLOv10  # type: ignore
from PIL import Image

def load_model():
    '''Load the YOLOv10 model for document layout analysis.'''
    return YOLOv10("/models/doclayout_yolo_docstructbench_imgsz1024.pt")

def predict_layout(model: YOLOv10, image: Image.Image, confidence: float):
    ''' Detect the layout of a document using the YOLOv10 model.

    Parameters
    ----------
    model : YOLOv10
        Model for document layout analysis.
    image : Image.Image
        The image to analyze.
    confidence : float
        Confidence threshold for detection.

    Returns
    -------
    _type_
        Detection results.
    '''
    results = model.predict(image, imgsz=1024, conf=confidence, device="cpu")[0]
    return results
