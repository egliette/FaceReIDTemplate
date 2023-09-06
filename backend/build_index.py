import glob
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
from annoy import AnnoyIndex
from retinaface import RetinaFace
from deepface import DeepFace
from deepface.commons import functions


model_name = "ArcFace"
image_folder = "faces/"
filename_path = "paths.txt"

model = DeepFace.build_model(model_name=model_name)
target_size = functions.find_target_size(model_name=model_name)

image_paths = glob.glob(image_folder + "*")
image_paths = list(sorted(image_paths))

with open(filename_path, "w") as file:
    for path in image_paths:
        file.write(path + "\n")

face_arrays = list()
embeddings = list()

for path in tqdm(image_paths):
    img = Image.open(path)
    img = np.array(img)
    face_array = RetinaFace.extract_faces(img_path=img, align=True)[0]
    face_array = cv2.resize(face_array, target_size)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = np.float32(face_array) / 255
    face_arrays.append(face_array)

new_face_arrays = np.concatenate(face_arrays, axis=0)

embeddings = np.float32(model.predict(new_face_arrays))


# Build Faiss Index
dimension = model.output_shape[1]
index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, "faiss.index")


# Build Annoy Index
f = model.output_shape[1]
t = AnnoyIndex(f, "euclidean")
for i, e in enumerate(embeddings):
    t.add_item(i, e)

t.build(10) # 10 trees
t.save('annoy.index')