FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY poses/ ./poses/

EXPOSE 8080

CMD ["python", "-m", "app.main"]
