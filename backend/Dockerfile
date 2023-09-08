FROM python:3.8.10-slim

WORKDIR /app

COPY app.py build_index.py requirements.txt ./

COPY faces/ /app/faces/

RUN apt-get update && \
    apt-get install gcc g++ -y && \
    pip install -r requirements.txt && \
    apt-get install ffmpeg libsm6 libxext6 -y

CMD python build_index.py && \
    uvicorn app:app --reload --port 8000 --host 0.0.0.0
