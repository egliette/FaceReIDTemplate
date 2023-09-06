import time
import base64
import io
from io import BytesIO

import cv2
import requests
import numpy as np
from PIL import Image
import streamlit as st


def query_image(image):
    img = image
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



st.set_page_config(
    page_title="Realtime Detection",
    page_icon="⏺️"
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



st.title("Livestream Upload")

# Open a connection to the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:

    ret, frame = cap.read()

    image1 = Image.fromarray(frame)
    results = query_image(image1)["results"]
    face_locations = [r["face_location"] for r in results]
    best_labels = [r["labels"][0] for r in results]
    best_dists = [r["distances"][0] for r in results]
    image2 = draw_bboxes(image1, face_locations, best_labels, best_dists)

    if not ret:
        break

    frame_placeholder.image(image2, channels="BGR")

    if stop_button_pressed:
        break


cap.release()