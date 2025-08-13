FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    swig \
    libopenblas-dev \
    libomp-dev \
    libfaiss-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r my-llm-helper/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "my-llm-helper.api:app", "--host", "0.0.0.0", "--port", "8000"]
