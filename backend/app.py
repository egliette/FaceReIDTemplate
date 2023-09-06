import os
import io
import base64
from datetime import datetime

import cv2
import faiss
import ffmpeg
import numpy as np
from tqdm import tqdm
from PIL import Image
from annoy import AnnoyIndex
from deepface import DeepFace
from deepface.commons import functions
from retinaface import RetinaFace
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File, Form


app = FastAPI()
image_folder = "faces/"
filename_path = "paths.txt"
model_name = "ArcFace"
model = DeepFace.build_model(model_name=model_name)
target_size = functions.find_target_size(model_name=model_name)

def load_data():
    faiss_index = faiss.read_index("faiss.index")
    annoy_index = AnnoyIndex(model.output_shape[1], "euclidean")
    annoy_index.load("annoy.index")

    return faiss_index, annoy_index

def crop_image_and_get_embeddings(img, model):
    results = RetinaFace.extract_faces(img_path=img, align=True)
    embeddings = list()
    for array in results:
        face_array = cv2.resize(array, target_size)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = np.float32(face_array) / 255
        embedding = np.float32(model.predict(face_array, verbose = 0))
        embeddings.append(embedding)

    return embeddings

def generate_unique_filename(dir=".", ext=""):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ext
    path = "/".join([dir, filename])
    while os.path.exists(path):
        path += "0"
    return path

def draw_bboxes(image, face_locations, labels, dists):

    img = np.array(image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    box_thickness = 2

    for i, location in enumerate(face_locations):

        top, right, bottom, left = location

        if labels[i] == "Unknown":
            box_color = (0, 0, 255)
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



@app.get("/")
async def read_root():
    return {"message": "Welcome!"}

@app.post("/add_image/")
async def add_image(file: UploadFile = File(...),
                    filename: str = Form(...)):
    # Read and save new image
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    # img = cv.imdecode(io.BytesIO(image_data), cv.IMREAD_COLOR)
    image_dest = image_folder + filename
    img.save(image_dest, format="PNG")

    with open(filename_path, "a") as file:
      file.write(image_dest + "\n")

    # Get face from image
    faiss_index, annoy_index = load_data()

    img = np.array(img)
    new_embedding = crop_image_and_get_embeddings(img, model)[0]

    faiss.normalize_L2(new_embedding)

    # Update Spotify Annoy Index
    f = new_embedding.shape[1]
    t = AnnoyIndex(f, "euclidean")
    for i in range(annoy_index.get_n_items()):
        embedding = annoy_index.get_item_vector(i)
        t.add_item(i, embedding)

    i = t.get_n_items()
    t.add_item(i, new_embedding[0])

    t.build(10) # 10 trees
    t.save("annoy.index")

    # Update Faiss Index
    faiss_index.add(new_embedding)
    faiss.write_index(faiss_index, "faiss.index")

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...),
                        k: int = Form(...),
                        search_type: str = Form(...)):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    img = np.array(img)

    results = list()

    face_locations = RetinaFace.detect_faces(img_path=img)
    face_locations = [location["facial_area"] for location in face_locations.values()]
    embeddings = crop_image_and_get_embeddings(img, model)

    faiss_index, annoy_index = load_data()

    for location, embedding in zip(face_locations, embeddings):

        faiss.normalize_L2(embedding)

        if search_type == "faiss":
            distances, neighbors = faiss_index.search(embedding, k)
            distances = [round(d, 2) for d in distances[0].tolist()]
            neighbors = neighbors[0].tolist()
        else:
            neighbors, distances = annoy_index.get_nns_by_vector(embedding[0], k, include_distances=True)
            distances = [round(1 - (d**2)/2 , 2) for d in distances]

        image_paths = list()
        with open(filename_path, "r") as file:
            image_paths = [line.strip() for line in file]

        labels = [image_paths[n].split("/")[-1] for n in neighbors]
        neighbor_paths = [image_paths[n] for n in neighbors]
        image_list = []
        for image_path in neighbor_paths:
            with open(image_path, "rb") as f:
                image = f.read()
                image_base64 = base64.b64encode(image).decode("utf-8")
                image_list.append(image_base64)

        left, top, right, bottom = [l.item() for l in location]

        results.append({
            "face_location": [top, right, bottom, left],
            "images": image_list,
            "distances": distances,
            "labels": labels,
        })

    return {"results": results}


@app.post("/process_video/")
async def process_video(video: UploadFile = File(...),
                        k: int = Form(...),
                        threshold: float = Form(...),
                        search_type: str = Form(...)):
    video_bytes = await video.read()

    input_video_path = generate_unique_filename(ext=".mp4")
    with open(input_video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    output_video_path = generate_unique_filename(ext=".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faiss_index, annoy_index = load_data()

    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = RetinaFace.detect_faces(img_path=frame)
        face_locations = [location["facial_area"] for location in face_locations.values()]
        face_labels = list()
        dists = list()

        for i, location in enumerate(face_locations):
            left, top, right, bottom = [l.item() for l in location]
            face_locations[i] = [top, right, bottom, left]
            embeddings = crop_image_and_get_embeddings(frame, model)
            embedding = embeddings[0]
            faiss.normalize_L2(embedding)

            if search_type == "faiss":
                distances, neighbors = faiss_index.search(embedding, k)
                distances = [round(d, 2) for d in distances[0].tolist()]
                neighbors = neighbors[0].tolist()
            else:
                neighbors, distances = annoy_index.get_nns_by_vector(embedding[0], k, include_distances=True)
                distances = [round(1 - (d**2)/2 , 2) for d in distances]

            image_paths = list()
            with open(filename_path, "r") as file:
                image_paths = [line.strip() for line in file]

            labels = [image_paths[n].split("/")[-1] for n in neighbors]
            neighbor_paths = [image_paths[n] for n in neighbors]
            image_list = []
            for image_path in neighbor_paths:
                with open(image_path, "rb") as f:
                    image = f.read()
                    image_base64 = base64.b64encode(image).decode("utf-8")
                    image_list.append(image_base64)

            dists.append(distances[0])

            if (distances[0] > threshold):
                face_labels.append(labels[0])
            else:
                face_labels.append("Unknown")

        frame = draw_bboxes(frame, face_locations, face_labels, dists)

        out.write(frame)

    H264_output_video_path = generate_unique_filename(ext=".mp4")
    ffmpeg.input(output_video_path).output(H264_output_video_path, vcodec="libx264").run()

    with open(H264_output_video_path, "rb") as f:
        video_bytes = f.read()
    video_bytesio = io.BytesIO(video_bytes)

    os.remove(input_video_path)
    os.remove(output_video_path)
    os.remove(H264_output_video_path)

    return StreamingResponse(video_bytesio, media_type="video/avi")
