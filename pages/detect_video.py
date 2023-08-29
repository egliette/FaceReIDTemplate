import streamlit as st
import requests


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

if uploaded_file:
    with st.spinner("Sending request..."):
        video = query_movie(uploaded_file)
        st.video(video)