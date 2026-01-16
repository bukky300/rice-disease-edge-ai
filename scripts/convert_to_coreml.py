

#!/usr/bin/env python3
import coremltools as ct
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SAVED_MODEL_DIR = "models/coreml_ready/mobilenetv3small_single"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/coreml")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Converting from:", SAVED_MODEL_DIR)

mlmodel = ct.convert(
    SAVED_MODEL_DIR,
    source="tensorflow",
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
)

out_path = os.path.join(
    OUTPUT_DIR,
    "RiceDiseaseClassifier_fp32.mlpackage"
)

mlmodel.save(out_path)

print("Saved CoreML model to:", out_path)

print("\nInput description:")
print(mlmodel.input_description)

print("\nOutput description:")
print(mlmodel.output_description)
