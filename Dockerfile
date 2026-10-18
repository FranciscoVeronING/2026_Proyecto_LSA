FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY poses/ ./poses/

EXPOSE 8080

CMD ["sh", "-c", "python main.py"]
