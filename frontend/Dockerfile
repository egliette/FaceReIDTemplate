FROM python:3.8.10-slim

WORKDIR /app

COPY requirements.txt main.py ./
copy pages/ /app/pages

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install -r requirements.txt

CMD streamlit run main.py --server.port 8050
