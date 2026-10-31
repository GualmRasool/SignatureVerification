from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from skimage.feature import hog
import numpy as np
import cv2
import json
import os
import uvicorn

app = FastAPI()

IMG_SIZE = 224
MATCH_THRESHOLD = 80.0  # percent - below this = "do not match"

# --- Load JSON reference map (same as before - no schema change) ---
with open("data/reference_data.json", "r") as f:
    reference_map = json.load(f)


def preprocess_signature_bytes(image_bytes):
    """
    1. Decode to grayscale
    2. Otsu threshold -> isolate ink from paper/background
    3. Crop tightly to the ink's bounding box -> ignore blank space
    4. Resize preserving aspect ratio, pad onto a square canvas -> no stretch
    Returns a binary image: ink=255, background=0
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not decode image bytes")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binary = binary[y:y + h, x:x + w]

    h, w = binary.shape
    scale = min((IMG_SIZE * 0.85) / w, (IMG_SIZE * 0.85) / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def extract_features(binary_img):
    """
    HOG: captures stroke direction/shape patterns - the main signal for
         telling different handwriting apart.
    Hu moments: shape descriptor invariant to translation/scale/rotation -
         a secondary signal for overall signature silhouette.
    Density: rough ink-coverage ratio - a weak tie-breaker signal.
    """
    hog_feat = hog(
        binary_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    density = np.sum(binary_img > 0) / binary_img.size

    return hog_feat, hu_log, density


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compare_signatures(ref_bytes, input_bytes):
    ref_binary = preprocess_signature_bytes(ref_bytes)
    input_binary = preprocess_signature_bytes(input_bytes)

    hog1, hu1, dens1 = extract_features(ref_binary)
    hog2, hu2, dens2 = extract_features(input_binary)

    hog_sim = cosine_sim(hog1, hog2)
    hu_sim = 1.0 / (1.0 + np.linalg.norm(hu1 - hu2))
    density_sim = 1.0 - min(abs(dens1 - dens2) / max(dens1, dens2, 1e-6), 1.0)

    # Weighted combination - HOG (stroke shape) carries most of the signal
    final = 0.80 * hog_sim + 0.15 * hu_sim + 0.05 * density_sim
    return round(final * 100, 2)


@app.get("/")
def home():
    return {"message": "Signature verification API (shape-feature based, local) running!"}


@app.post("/compare")
async def compare_signature(name: str = Form(...), file: UploadFile = File(...)):
    # --- Same request contract as before: no .NET changes needed ---
    if name not in reference_map:
        return JSONResponse({"error": f"No reference found for {name}"}, status_code=404)

    ref_path = reference_map[name]
    if not os.path.exists(ref_path):
        return JSONResponse({"error": f"Reference image missing at {ref_path}"}, status_code=404)

    ref_bytes = open(ref_path, "rb").read()
    input_bytes = await file.read()

    try:
        match_percent = compare_signatures(ref_bytes, input_bytes)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Cast numpy scalar types to native Python types - FastAPI's JSON
    # encoder cannot serialize numpy.float64 / numpy.bool_ directly.
    match_percent = float(match_percent)
    is_match = bool(match_percent >= MATCH_THRESHOLD)
    result = "✅ Signature Successfully Matched" if is_match else "❌ Signature Does Not Match"

    return {
        "name": name,
        "match_percent": match_percent,
        "threshold": MATCH_THRESHOLD,
        "is_match": is_match,
        "result": result
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))