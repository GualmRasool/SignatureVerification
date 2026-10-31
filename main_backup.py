from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import joblib
import json
import os
from io import BytesIO
import uvicorn

app = FastAPI()

# --- Load models ---
clf = joblib.load("models/signature_verification_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# --- Load JSON reference map ---
with open("data/reference_data.json", "r") as f:
    reference_map = json.load(f)

# --- Load feature extractor ---
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_deep_features_from_bytes(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return feature_model.predict(x).flatten()

@app.get("/")
def home():
    return {"message": "Signature verification API running on Render!"}

@app.post("/compare")
async def compare_signature(name: str = Form(...), file: UploadFile = File(...)):
    if name not in reference_map:
        return JSONResponse({"error": f"No reference found for {name}"}, status_code=404)

    ref_path = reference_map[name]
    if not os.path.exists(ref_path):
        return JSONResponse({"error": f"Reference image missing at {ref_path}"}, status_code=404)

    ref_bytes = open(ref_path, "rb").read()
    input_bytes = await file.read()

    ref_feat = extract_deep_features_from_bytes(ref_bytes)
    input_feat = extract_deep_features_from_bytes(input_bytes)

    pred_ref = clf.predict([ref_feat])[0]
    pred_input = clf.predict([input_feat])[0]
    sim = float(cosine_similarity([ref_feat], [input_feat])[0][0])
    percent = round(sim * 100, 2)

    if pred_ref == pred_input and sim >= 0.85:
        result = "✅ Signatures Match"
    elif pred_ref == pred_input:
        result = "⚠️ Same user predicted, low similarity"
    else:
        result = "❌ Signatures Do Not Match"

    return {
        "reference_user": str(pred_ref),
        "input_user": str(pred_input),
        "similarity": sim,
        "match_percent": percent,
        "result": result
    }

# --- Run locally ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
