FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md config.example.toml ./
COPY stratcheck ./stratcheck
COPY scripts ./scripts
COPY configs ./configs
COPY data ./data

RUN python -m pip install --upgrade pip \
    && pip install .

CMD ["python", "scripts/docker_runner.py"]
