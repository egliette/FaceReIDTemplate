import os
import time
import base64
import io
from io import BytesIO
from datetime import datetime

import cv2
import ffmpeg
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import streamlit as st


st.set_page_config(
    page_title="Video Detection",
    page_icon="🎥"
)

if "url" not in st.session_state:
    st.session_state["url"] = "http://localhost:XXXX"
if "k" not in st.session_state:
    st.session_state["k"] = 3
if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5
if "search_type" not in st.session_state:
    st.session_state["search_type"] = "faiss"

st.sidebar.text_input("API URL", 
                     value=st.session_state["url"], 
                     key="url")

st.sidebar.slider("Max neighbors", 
                    min_value=1, 
                    max_value=10,
                    step=1, 
                    value=st.session_state["k"],
                    key="k")

st.sidebar.number_input("Cosine Similarity Threshold", 
                        min_value=-1.0, 
                        max_value=1.0,
                        step=0.01, 
                        value=st.session_state["threshold"],
                        key="threshold")

search_types = ["faiss", "spotify_annoy"]
st.sidebar.radio("Select search type", 
                 search_types,
                 index=search_types.index(st.session_state["search_type"]),
                 key="search_type")


def query_image(image):
    img = Image.fromarray(image)
    image_api_url = st.session_state["url"] + "/process_image/"

    image_bytes = BytesIO()
    img.save(image_bytes, format="JPEG")

    files = {"file": image_bytes.getvalue()}
    data = {"k": st.session_state["k"], 
            "search_type": st.session_state["search_type"]}

    response = requests.post(image_api_url, files=files, data=data)

    results = response.json()

    return results

def draw_bboxes(image, face_locations, labels, dists):

    img = np.array(image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    box_thickness = 2

    for i, location in enumerate(face_locations):

        top, right, bottom, left = location
        
        if dists[i] < st.session_state["threshold"]:
            box_color = (0, 0, 255)
            labels[i] = "Unknown"
        else:
            box_color = (0, 255, 0)

        font_color = (255, 255, 255)

        cv2.rectangle(img, (left, top), (right, bottom), box_color, box_thickness)
        
        label = " ".join([labels[i], str(dists[i])])
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        bg_rect_top_left = (left - 5, top - text_size[1] - 15)
        bg_rect_bottom_right = (left + text_size[0] + 5, top - 5)

        cv2.rectangle(img, bg_rect_top_left, bg_rect_bottom_right, box_color, -1)
        cv2.putText(img, label, (left, top-10), font, font_scale, font_color, font_thickness)

    return img

def generate_unique_filename(dir="temp", ext=""):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ext
    path = "/".join([dir, filename])
    while os.path.exists(path):
        path += "0"
    return path

def query_movie(video):
    files = {"video": video}
    data = {"k": st.session_state["k"],
            "threshold": st.session_state["threshold"],
            "search_type": st.session_state["search_type"]}
    
    video_api_url = st.session_state["url"] + "/process_video/"

    response = requests.post(video_api_url, files=files, data=data)

    return response.content


st.title("Detect Video")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mkv"])

col1, col2 = st.columns(2)
with col1:
    image_mode = st.button("Progess each image 🐢")
with col2:
    video_mode = st.button("Progress whole video 🐇")

if uploaded_file and image_mode:
    input_video_path = generate_unique_filename(ext=".mp4")
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    output_video_path = generate_unique_filename(ext=".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0, text="Please wait...")

    for frame_num in tqdm(range(total_frames)):
        progress_bar.progress((frame_num+1)/total_frames, 
                              text=f"Frame: {frame_num+1}/{total_frames}")
        ret, frame = cap.read()
        if not ret:
            break

        results = query_image(frame)["results"]
        face_locations = [r["face_location"] for r in results]
        best_labels = [r["labels"][0] for r in results]
        best_dists = [r["distances"][0] for r in results]
        frame = draw_bboxes(frame, face_locations, best_labels, best_dists)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # current_image.image(frame)
        out.write(frame)

    H264_output_video_path = generate_unique_filename(ext=".mp4")
    ffmpeg.input(output_video_path).output(H264_output_video_path, vcodec="libx264").run()
    
    st.video(H264_output_video_path)

    os.remove(input_video_path)
    os.remove(output_video_path)
    os.remove(H264_output_video_path)
elif uploaded_file and video_mode:
    with st.spinner("Sending request..."):
        video = query_movie(uploaded_file)
        st.video(video)
elif (image_mode or video_mode) and uploaded_file is None:
    st.error("Please upload a video.")
