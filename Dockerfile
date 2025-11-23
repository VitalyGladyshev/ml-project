FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt -qqq && \
    pip install mlflow boto3 -qqq

# Копирование кода
COPY . .

# Экспозиция портов
EXPOSE 5000

# Команда запуска
CMD ["bash", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns & sleep 5 && python src/main.py"]
