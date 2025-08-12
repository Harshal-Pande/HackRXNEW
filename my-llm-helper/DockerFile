# Use official Python slim image
FROM python:3.13-slim

# Install system dependencies needed by faiss-cpu and building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    swig \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Upgrade pip and install python dependencies from your nested requirements.txt
RUN pip install --upgrade pip
RUN pip install -r my-llm-helper/requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "my-llm-helper.api:app", "--host", "0.0.0.0", "--port", "8000"]
