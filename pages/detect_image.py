import time
import base64
import io
from io import BytesIO

import cv2
import requests
import numpy as np
from PIL import Image
import streamlit as st


st.set_page_config(
    page_title="Image Detection",
    page_icon="📷"
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
            box_color = (255, 0, 0)
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


st.title("Detect Image")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if st.button("Submit"):
    if uploaded_image is not None:
        with st.spinner("Sending request..."):
            image1 = Image.open(uploaded_image)
        
            start = time.time()
            results = query_image(image1)["results"]
            request_time = round(time.time() - start, 2)

            face_locations = [r["face_location"] for r in results]
            best_labels = [r["labels"][0] for r in results]
            best_dists = [r["distances"][0] for r in results]
            
            image2 = draw_bboxes(image1, face_locations, best_labels, best_dists)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image1, caption="Source", use_column_width=True)
            with col2:
                st.image(image2, caption="Prediction", use_column_width=True)

            st.header(f"Predictions (~{request_time} seconds)")
            image1_np = np.array(image1)

            for prediction in results:
                location = prediction["face_location"]
                dists = prediction["distances"]
                labels = prediction["labels"]
                image_base64_list = prediction["images"]

                cols = st.columns(1 + st.session_state.k)
                top, right, bottom, left = location
                face_array = image1_np[top:bottom, left:right]

                with cols[0]:
                    st.image(face_array, caption="Source", use_column_width=True)
                
                for i, predicted_col in enumerate(cols[1:]):
                    if dists[i] >= st.session_state["threshold"]:
                        with predicted_col:
                            img = image_base64_list[i]
                            image_data = base64.b64decode(img)
                            predicted_image = Image.open(io.BytesIO(image_data))
                            caption = f"{labels[i]} {dists[i]}"
                            st.image(predicted_image, caption=caption, use_column_width=True)
    else:
        st.error("Please upload an image.app")
