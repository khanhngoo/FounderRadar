FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py data_pipeline.py ./
COPY scripts ./scripts
COPY data ./data

RUN python scripts/fetch_data.py || true

EXPOSE 3838

CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "3838"]
