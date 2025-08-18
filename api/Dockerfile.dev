FROM python:3.12-slim

WORKDIR /api

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    wget \
    libgl1-mesa-dri \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Check that libGL.so.1 is found
RUN ldconfig -p | grep libGL.so.1 || echo "libGL.so.1 not found"

COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /models && \
    wget -O /models/doclayout_yolo_docstructbench_imgsz1024.pt https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
