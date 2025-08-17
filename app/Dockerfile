# app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

COPY config/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /ScribaLLM/logs && \
    touch /ScribaLLM/logs/ocr.log && \
    touch /ScribaLLM/logs/chatbot.log && \
    touch /ScribaLLM/logs/main.log  && \
    export ANONYMIZED_TELEMETRY=False

COPY . .
RUN mkdir -p /root/.streamlit && cp -r config/.streamlit/* /root/.streamlit/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]