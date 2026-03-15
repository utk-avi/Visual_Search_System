import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def encode_image(query_image: Image.Image):

    image_input = preprocess(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features/image_features.norm(dim=-1, keepdim=True)

    search_image=image_features.cpu().numpy()

    search_image=search_image.tolist()

    return search_image