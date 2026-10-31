from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import uvicorn

app = FastAPI()


def process_image(img):
    if img is None:
        return None

    _, thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    h, w = img.shape
    scale = 128 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((128, 128), dtype=np.uint8) * 255
    x_offset = (128 - new_w) // 2
    y_offset = (128 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    img = canvas

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.astype(np.float32) / 255.0


def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return process_image(img)


def preprocess_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return process_image(img)


def l1_distance(vectors):
    x, y = vectors
    return tf.abs(x - y)


# --- Load the trained Siamese model ---
model = tf.keras.models.load_model(
    "signature_model.keras",
    custom_objects={"l1_distance": l1_distance},
    compile=False,
    safe_mode=False
)
print("MODEL LOADED SUCCESSFULLY")

# --- Reference map: name -> reference image path (same format your app already uses) ---
with open("data/reference_data.json", "r") as f:
    reference_map = json.load(f)
print("REFERENCE DATA LOADED")


@app.get("/")
def home():
    return {"message": "Signature Verification API (trained Siamese CNN) running locally!"}


@app.post("/compare")
async def compare_signature(name: str = Form(...), file: UploadFile = File(...)):
    # --- Same contract as before: no .NET changes needed ---
    if name not in reference_map:
        return JSONResponse({"error": f"No reference found for {name}"}, status_code=404)

    ref_entry = reference_map[name]
    # Support both a single path (string) and multiple paths (list) -
    # if multiple, use the first one found on disk (this model architecture
    # takes exactly ONE reference image per comparison - see note below
    # about multi-reference support)
    ref_paths = [ref_entry] if isinstance(ref_entry, str) else ref_entry
    ref_path = next((p for p in ref_paths if os.path.exists(p)), None)

    if ref_path is None:
        return JSONResponse({"error": f"Reference image missing for {name}"}, status_code=404)

    ref_img = preprocess(ref_path)
    if ref_img is None:
        return JSONResponse({"error": "Reference image could not be processed"}, status_code=400)

    input_bytes = await file.read()
    test_img = preprocess_bytes(input_bytes)
    if test_img is None:
        return JSONResponse({"error": "Invalid uploaded image"}, status_code=400)

    ref_img = np.expand_dims(ref_img, axis=0)
    test_img = np.expand_dims(test_img, axis=0)

    pred = model.predict([ref_img, test_img], verbose=0)[0][0]
    match_percent = round(float(pred) * 100, 2)

    if match_percent >= 85:
        result = "✅ Signature Successfully Matched"
        is_match = True
    elif match_percent >= 65:
        result = "⚠️ Borderline - please review manually"
        is_match = False
    else:
        result = "❌ Signature Does Not Match"
        is_match = False

    return {
        "name": name,
        "match_percent": match_percent,
        "is_match": is_match,
        "result": result
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))