FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY key.pem /app/key.pem 
COPY cert.pem /app/cert.pem

EXPOSE 8501

CMD ["streamlit", "run", "modelo_obs.py", "--server.port=8501", "--server.address=0.0.0.0"]


