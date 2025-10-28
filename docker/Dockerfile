# Dockerfile for VA-Prediction
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


WORKDIR /app
COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


CMD ["bash", "docker/entrypoint.sh"]
