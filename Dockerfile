# Use an official NVIDIA CUDA base image with Ubuntu
FROM python:3.12-slim as base

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser


COPY . /home/appuser

RUN chown -R appuser:appuser /home/appuser
USER appuser

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .
RUN python -m spacy download en_core_web_md && python -m spacy download de_core_news_md
EXPOSE 8000

CMD ["python", "/scripts/model_download.py"]