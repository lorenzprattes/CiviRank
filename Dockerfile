# Use an official NVIDIA CUDA base image with Ubuntu
FROM python:3.12-slim as base

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser


COPY . /home/appuser

RUN chown -R appuser:appuser /home/appuser
USER appuser

# RUN python -m venv /home/appuser/venv
# ENV PATH="/home/appuser/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "/scripts/model_download.py"]