FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY ui ./ui
COPY models ./models
COPY data ./data
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
