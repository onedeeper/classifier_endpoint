from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
from pydantic import BaseModel
from typing import Any
from PIL import Image
import requests
import uvicorn
import os
import pickle

app = FastAPI(title= " Bird or Bug Classifier")
@app.post('/predict/image_model/', status_code=200)
async def predict(file : UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    learn = load_learner('src/imgClassifier.pkl')
    try:
        img = PILImage.create(file.filename)
    except Exception as e:
        os.remove(file.filename)
        return {"status_code": 400,
                "message": "file could not be opened"
                }
    prediction, _, probs = learn.predict(img)
    os.remove(file.filename)
    return {"status_code": 200,
            "predicted_label": prediction,
            "probs": {probs[0].tolist(), probs[1].tolist()}
            }