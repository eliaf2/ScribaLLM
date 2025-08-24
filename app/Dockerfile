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

RUN git clone --depth 1 --branch main https://github.com/eliaf2/ScribaLLM.git /tmp/repo && \
    cp -r /tmp/repo/app/* /app/ && \
    rm -rf /tmp/repo

RUN pip3 install --no-cache-dir -r config/requirements.txt

RUN mkdir -p /ScribaLLM/logs && \
    touch /ScribaLLM/logs/ocr.log && \
    touch /ScribaLLM/logs/chatbot.log && \
    touch /ScribaLLM/logs/main.log && \
    export ANONYMIZED_TELEMETRY=False

RUN mkdir -p /root/.streamlit && cp -r config/.streamlit/* /root/.streamlit/
RUN mkdir -p /ScribaLLM/tmp && cp -r config/tmp/* /ScribaLLM/tmp/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]