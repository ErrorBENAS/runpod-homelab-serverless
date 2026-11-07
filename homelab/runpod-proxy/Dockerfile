FROM python:3.13-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
USER nobody
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
