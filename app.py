from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
import json
import io
import plotly.express as px
import plotly.graph_objects as go
IMG_HEIGHT = 180
IMG_WIDTH = 180

app = FastAPI()



if __name__ == "__main__":
    uvicorn.run(app, reload=True, host='0.0.0.0', port=8000)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod: ["http://127.0.0.1:5500", "http://localhost:5500", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def build_model(num_classes: int):
    return Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes),
    ])


model_age = build_model(len(CLASS_NAMES))
model_age.load_weights(r"models\age_model_weights.weights.h5")
model_gender = build_model(2)
model_gender.load_weights('models/gender_model.h5')

def _try_parse_ages(class_names):
    try:
        return np.array([int(c) for c in class_names], dtype=np.float32)
    except Exception:
        return None
    
AGES = _try_parse_ages(CLASS_NAMES)

def preprocess_image_bytes(image_bytes: bytes) -> tf.Tensor:
    if not image_bytes:
        raise ValueError("Empty image bytes")

    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

predicted_ages = {str(c): 0 for c in CLASS_NAMES}
predicted_gender = {'female': 0, 'male': 0}

@app.get('/')
async def root():
    return {'message': 'Healthy'}

@app.post('/predict/age')
async def preidct_age(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(status_code=415, detail=f'Unsupported file type: {file.content_type}')

    image_bytes = await file.read()

    try:
        x = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    logits = model_age.predict(x, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    response={
        'predicted_class': pred_class,
        'confidence': conf,
    }

    if AGES is not None:
        expected_age = float(np.sum(probs * AGES))
        predicted_age = int(AGES[pred_idx])
        response.update({
            "predicted_age": predicted_age,
            "expected_age": expected_age,
        })
    predicted_ages[pred_class] += 1
   
    return response

@app.post('/predict/gender')
async def predict_gender(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(status_code=415, detail=f'Unsupported file type: {file.content_type}')
    
    image_bytes = await file.read()

    try:
        x=preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    logits = model_gender.predict(x, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = 'female' if pred_idx == 0 else 'male'
    conf=float(probs[pred_idx])

    response = {
        'predicted_gender': pred_class,
        'confidence': conf
    }
    predicted_gender[pred_class] += 1

    

    return response

@app.get('/plotly/age')
async def plot_age():
    x=list(predicted_ages.keys())
    y=list(predicted_ages.values())

    fig = go.Figure(
        data=[go.Bar(x=x, y=y)],
        layout=go.Layout(
            title="Age",
            xaxis_title="Predicted Age",
            yaxis_title ="Count",
            margin=dict(l=40, r=20, t=50, b=40)
        ),
    )

    return JSONResponse(content=fig.to_dict())

@app.get('/plotly/gender')
async def plot_gender():
    x=list(predicted_gender.keys())
    y=list(predicted_gender.values())

    fig = go.Figure(
        data=[go.Bar(x=x, y=y)],
        layout=go.Layout(
            title="Age",
            xaxis_title="Predicted Gender",
            yaxis_title ="Count",
            margin=dict(l=40, r=20, t=50, b=40)
        ),
    )

    return JSONResponse(content=fig.to_dict())


    

    