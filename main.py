from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from clip_encoder import encode_image
from sim import compute_similarity

app = FastAPI()

@app.get('/')
def home():
    return {"Message":"Visual Search System"}

@app.post('/upload_image')
async def upload(file: UploadFile=File(...)):

    contents =await file.read()

    query_image = Image.open(io.BytesIO(contents)).convert("RGB")

    encoding = encode_image(query_image)
    
    results = compute_similarity(encoding)

    return results
