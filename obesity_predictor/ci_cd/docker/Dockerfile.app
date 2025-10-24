# Dockerfile for Streamlit Dashboard
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

COPY obesity_predictor /app/obesity_predictor
COPY data /app/data
EXPOSE 8501

CMD ["streamlit", "run", "obesity_predictor/app/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]