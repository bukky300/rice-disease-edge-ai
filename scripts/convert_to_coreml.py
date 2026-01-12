# import coremltools as ct
# from coremltools.models.neural_network import quantization_utils
# import numpy as np
# from PIL import Image
# import os

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# SAVED_MODEL_DIR = os.path.join(
#     PROJECT_ROOT, "models/coreml_ready/mobilenetv2_rice"
# )

# OUT_DIR = os.path.join(PROJECT_ROOT, "models/coreml")
# os.makedirs(OUT_DIR, exist_ok=True)

# OUT_PATH = os.path.join(
#     OUT_DIR, "mobilenetv2_rice_int8.mlmodel"
# )

# LABELS = [
#     "Bacterial Leaf Blight",
#     "Brown Spot",
#     "Healthy Rice Leaf",
#     "Leaf Blast",
#     "Leaf Scald",
#     "Sheath Blight",
# ]

# print("Converting SavedModel → CoreML (FP32)")

# mlmodel_fp32 = ct.convert(
#     SAVED_MODEL_DIR,
#     source="tensorflow",
#     convert_to="neuralnetwork",
#     inputs=[
#         ct.ImageType(
#             name="inputs",
#             shape=(1, 160, 160, 3),
#             scale=1 / 127.5,
#             bias=[-1.0, -1.0, -1.0],
#         )
#     ],
#     minimum_deployment_target=ct.target.iOS13,
# )

# print("Quantizing weights → INT8")

# mlmodel_int8 = quantization_utils.quantize_weights(
#     mlmodel_fp32,
#     nbits=8,
#     quantization_mode="linear",
# )

# mlmodel_int8.save(OUT_PATH)

# print("Saved:", OUT_PATH)
# print("Model size (MB):", round(os.path.getsize(OUT_PATH) / 1e6, 2))


import coremltools as ct
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SAVED_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models/coreml_ready/mobilenetv2_rice"
)

OUT_DIR = os.path.join(PROJECT_ROOT, "models/coreml")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUT_DIR, "mobilenetv2_rice_int8.mlpackage"
)

print("Converting → CoreML MLProgram (INT8, simulator-safe)")

mlmodel = ct.convert(
    SAVED_MODEL_DIR,
    source="tensorflow",
    convert_to="mlprogram",
    inputs=[
        ct.ImageType(
            name="inputs",
            shape=(1, 160, 160, 3),
            scale=1 / 127.5,
            bias=[-1.0, -1.0, -1.0],
        )
    ],
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.iOS16,
)

# 🔑 MLProgram INT8 quantization (correct way)
mlmodel = ct.optimize.coreml.quantization.linear_quantize_weights(
    mlmodel,
    nbits=8
)

mlmodel.save(OUT_PATH)

print("✓ Saved:", OUT_PATH)
print("✓ Size (MB):", round(os.path.getsize(OUT_PATH) / 1e6, 2))
