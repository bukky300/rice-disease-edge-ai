import coremltools as ct
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SAVED_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models/coreml_ready/mobilenetv2_rice"
)

OUT_PATH = os.path.join(
    PROJECT_ROOT, "models/coreml/mobilenetv2_rice.mlpackage"
)

mlmodel = ct.convert(
    SAVED_MODEL_DIR,
    source="tensorflow",
    convert_to="mlprogram",
    inputs=[ct.ImageType(
        shape=(1, 160, 160, 3),
        scale=1/127.5,
        bias=[-1, -1, -1]
    )],
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save(OUT_PATH)

print("Saved CoreML model to:", OUT_PATH)
