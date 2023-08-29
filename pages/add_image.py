from io import BytesIO

import requests
from PIL import Image
import streamlit as st


st.set_page_config(
    page_title="Add Image to Dataset",
    page_icon="💾"
)


if "url" not in st.session_state:
    st.session_state["url"] = "http://localhost:XXXX"

st.sidebar.text_input("API URL", 
                     value=st.session_state["url"], 
                     key="url")

def add_image(image, filename):

    img = image
    image_api_url = st.session_state["url"] + "/add_image/"

    image_bytes = BytesIO()
    img.save(image_bytes, format="JPEG")

    files = {"file": image_bytes.getvalue()}
    data = {"filename": filename}

    response = requests.post(image_api_url, files=files, data=data)

    return response.status_code


st.title("Add Images to Dataset")

uploaded_images = st.file_uploader("Upload Image", 
                                    type=["jpg", "png", "jpeg"], 
                                    accept_multiple_files=True)

if st.button("Submit"):
    if uploaded_images is not None:
        with st.spinner("Sending request..."):
            for image_file in uploaded_images:
                image = Image.open(image_file)
                filename = image_file.name
                status_code = add_image(image, filename)
                
                if status_code == 200:
                    st.success(f"Add image {filename} successfully!")
                else:
                    st.error(f"Error: status code {status_code} at image {filename}")
    else:
        st.error("Please upload an image.")

