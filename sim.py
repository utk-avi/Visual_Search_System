import faiss
import pickle
import numpy as np

def compute_similarity(image_embedding, k:int=10):

    index = faiss.read_index("image_index.faiss")
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)

    image_embedding=np.array(image_embedding).astype("float32")

    distance, indices = index.search(image_embedding, k)

    results=[]

    for score, idx in zip(distance[0], indices[0]):
        results.append({
            "image_path":image_paths[idx],
            "similarity_score": float(score)
        })

    
    return results

