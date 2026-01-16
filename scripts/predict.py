import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

import os
import tensorflow as tf


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "baseline",
    "mobilenetv3small_infer",
    "rice_model_baseline.keras",
)

IMG_SIZE = (160, 160)

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
]
model = tf.keras.models.load_model(MODEL_PATH)


image_path = sys.argv[1]
if not os.path.isabs(image_path):
    image_path = os.path.join(PROJECT_ROOT, image_path)

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# ---- preprocess ----
img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
x = np.array(img).astype(np.float32)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = x[None, ...]

# ---- predict ----
pred = model.predict(x)[0]
idx = np.argmax(pred)

print(class_names[idx], float(pred[idx]))
