from doclayout_yolo import YOLOv10  # type: ignore

def load_model():
    return YOLOv10("/models/doclayout_yolo_docstructbench_imgsz1024.pt")

def predict_layout(model, image, confidence):
    results = model.predict(image, imgsz=1024, conf=confidence, device="cpu")[0]
    return results
