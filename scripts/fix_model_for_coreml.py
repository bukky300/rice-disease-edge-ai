import tensorflow as tf
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOURCE_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models/baseline/mobilenetv2_rice"
)

EXPORT_DIR = os.path.join(
    PROJECT_ROOT, "models/coreml_ready/mobilenetv2_rice"
)

os.makedirs(EXPORT_DIR, exist_ok=True)

# Load inference-only layer
model = tf.keras.layers.TFSMLayer(
    SOURCE_MODEL_DIR,
    call_endpoint="serving_default",
)


inputs = tf.keras.Input(shape=(160, 160, 3), dtype=tf.float32)
outputs = model(inputs)
clean_model = tf.keras.Model(inputs, outputs)


tf.saved_model.save(clean_model, EXPORT_DIR)

print("CoreML-ready SavedModel exported to:", EXPORT_DIR)
