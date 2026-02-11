FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV HF_HOME=/opt/hf
ENV TRANSFORMERS_CACHE=/opt/hf/transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/hf /opt/hf/transformers

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    flask \
    werkzeug \
    numpy \
    faiss-cpu \
    sentence-transformers \
    rank-bm25 \
    transformers \
    openai \
    gunicorn \
    bitsandbytes \
    accelerate \
    SQLAlchemy \
    psycopg2-binary


RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM"

COPY . /app

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "--timeout", "6000", "--graceful-timeout", "6000", "-b", "0.0.0.0:8080", "server:app"]
