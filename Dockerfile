FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-test.txt

EXPOSE 8000

COPY . .

CMD ["pytest", "-q"]
