# Use an official NVIDIA CUDA base image with Ubuntu
FROM python:3.12-slim as base

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

RUN pip install --upgrade pip

COPY . .

RUN chown -R appuser:appuser /home/appuser

USER appuser

RUN pip install -e .

EXPOSE 8000

CMD ["python", "/scripts/model_download.py"]