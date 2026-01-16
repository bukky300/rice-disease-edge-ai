# #!/usr/bin/env python3
# """
# Prepare model for CoreML conversion by creating a clean SavedModel
# """

# import tensorflow as tf
# import os


# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # Load the .keras file
# KERAS_MODEL = os.path.join(PROJECT_ROOT, "models/baseline/mobilenetv3small_infer/rice_model_baseline.keras")
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/coreml_ready/mobilenetv3small")

# print("="*60)
# print("Preparing Model for CoreML Conversion")
# print("="*60)
# print(f"Input: {KERAS_MODEL}")
# print(f"Output: {OUTPUT_DIR}")

# # Load model
# print("\n[1/2] Loading model...")
# model = tf.keras.models.load_model(KERAS_MODEL)
# print(f"✓ Model loaded")
# print(f"  Input: {model.input_shape}")
# print(f"  Output: {model.output_shape}")



# # Save as clean SavedModel with single signature
# print("\n[2/2] Saving as CoreML-ready SavedModel...")

# # Create a concrete function with fixed input signature
# # @tf.function(input_signature=[tf.TensorSpec(shape=[1, 160, 160, 3], dtype=tf.float32)])
# # def serve(inputs):
# #     return model(inputs, training=False)

# # Save with single signature
# # tf.saved_model.save(
# #     model,
# #     OUTPUT_DIR,
# #     signatures={'serving_default': serve}
# # )

# model.save(OUTPUT_DIR, save_format="tf")

# print(f"✓ Saved to: {OUTPUT_DIR}")
# print(f"\n✅ Model is now ready for CoreML conversion!")
# print(f"   Run: python scripts/convert_to_coreml.py")

#!/usr/bin/env python3
import tensorflow as tf
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "models/baseline/mobilenetv3small"
)

DST_DIR = os.path.join(
    PROJECT_ROOT,
    "models/coreml_ready/mobilenetv3small_single"
)

# clean output
if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)

print("Loading SavedModel...")
loaded = tf.saved_model.load(SRC_DIR)

infer = loaded.signatures["serving_default"]

print("Saving single-signature model...")
tf.saved_model.save(
    loaded,
    DST_DIR,
    signatures={"serving_default": infer}
)

print("✅ Single-signature SavedModel written to:")
print(DST_DIR)
