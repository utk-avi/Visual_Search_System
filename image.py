import os
import pickle

import clip
import faiss
import numpy as np
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "dataset"
batch_size = 5
supported_extensions = (".jpg", ".jpeg", ".png")

embeddings = []
image_paths = []
batch_images = []
batch_paths = []


def is_supported_image(filename):
    return filename.lower().endswith(supported_extensions)


def flush_batch():
    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        batch_embeddings = model.encode_image(batch_tensor)

    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

    embeddings.append(batch_embeddings.cpu().numpy())
    image_paths.extend(batch_paths)

    batch_images.clear()
    batch_paths.clear()


# walk through folder
for root, _, files in os.walk(image_folder):
    for file_name in files:

        if not is_supported_image(file_name):
            continue

        path = os.path.join(root, file_name)

        try:
            image = Image.open(path).convert("RGB")
        except:
            continue

        batch_images.append(preprocess(image))
        batch_paths.append(path)

        if len(batch_images) == batch_size:
            flush_batch()


# flush remaining images
if batch_images:
    flush_batch()

if not embeddings:
    raise ValueError("No supported images found in dataset folder")

embeddings = np.vstack(embeddings)

dimension = embeddings.shape[1]

# create FAISS index
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# save index
faiss.write_index(index, "image_index.faiss")

# save paths
with open("image_paths.pkl", "wb") as f:
    pickle.dump(image_paths, f)
