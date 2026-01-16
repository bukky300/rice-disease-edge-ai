
# #!/usr/bin/env python3
# """
# Convert TensorFlow SavedModel to CoreML for iOS deployment
# """

# import coremltools as ct
# import os

# # ============================================================
# # Configuration
# # ============================================================
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # Use the CoreML-ready SavedModel directory
# SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/coreml_ready/mobilenetv3small")
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/coreml")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# CLASS_LABELS = [
#     "Bacterial Leaf Blight",
#     "Brown Spot",
#     "Healthy Rice Leaf",
#     "Leaf Blast",
#     "Leaf Scald",
#     "Sheath Blight"
# ]

# print("="*60)
# print("CoreML Conversion for iOS")
# print("="*60)
# print(f"Input: {SAVED_MODEL_DIR}")
# print(f"Output: {OUTPUT_DIR}")

# # ============================================================
# # Convert to CoreML FP32
# # ============================================================
# print("\n[1/3] Converting to CoreML FP32...")

# mlmodel_fp32 = ct.convert(
#     SAVED_MODEL_DIR,
#     source="tensorflow",
#     convert_to="mlprogram",
#     compute_units=ct.ComputeUnit.ALL,
#     minimum_deployment_target=ct.target.iOS15,
# )

# mlmodel_fp32.author = "Chukwuebuka Emmanuel Igbokweuche"
# mlmodel_fp32.short_description = "Rice disease classifier (FP32)"
# mlmodel_fp32.version = "1.0"

# fp32_path = os.path.join(OUTPUT_DIR, "RiceDiseaseClassifier_fp32.mlpackage")
# mlmodel_fp32.save(fp32_path)
# fp32_size = os.path.getsize(fp32_path) / (1024 * 1024) if os.path.isfile(fp32_path) else sum(
#     f.stat().st_size for f in os.scandir(fp32_path) if f.is_file()
# ) / (1024 * 1024)

# print(f"✓ Saved: {fp32_size:.2f} MB")
# print(f"  Path: {fp32_path}")

# # ============================================================
# # Convert to CoreML FP16
# # ============================================================
# print("\n[2/3] Converting to CoreML FP16...")

# mlmodel_fp16 = ct.convert(
#     SAVED_MODEL_DIR,
#     source="tensorflow",
#     convert_to="mlprogram",
#     compute_units=ct.ComputeUnit.ALL,
#     compute_precision=ct.precision.FLOAT16,
#     minimum_deployment_target=ct.target.iOS15,
# )
# mlmodel_fp16.author = "Chukwuebuka Emmanuel Igbokweuche"
# mlmodel_fp16.short_description = "Rice disease classifier (FP16)"
# mlmodel_fp16.version = "1.0"

# fp16_path = os.path.join(OUTPUT_DIR, "RiceDiseaseClassifier_fp16.mlpackage")
# mlmodel_fp16.save(fp16_path)

# from pathlib import Path
# fp16_size = sum(f.stat().st_size for f in Path(fp16_path).rglob('*') if f.is_file()) / (1024 * 1024)

# print(f"✓ Saved: {fp16_size:.2f} MB")
# print(f"  Path: {fp16_path}")

# # ============================================================
# # Convert to CoreML INT8 (Quantized)
# # ============================================================
# print("\n[3/3] Converting to CoreML INT8...")

# mlmodel_int8 = ct.convert(
#     SAVED_MODEL_DIR,
#     source="tensorflow",
#     convert_to="mlprogram",
#     compute_units=ct.ComputeUnit.ALL,
#     minimum_deployment_target=ct.target.iOS15,
# )

# # Apply INT8 quantization
# print("  Applying INT8 quantization...")
# # 1. Define the operator-level config (8-bit symmetric is standard for Neural Engine)
# op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
#     mode="linear_symmetric", 
#     weight_threshold=512  # Threshold for minimum number of elements to quantize
# )

# # 2. Wrap it in a global OptimizationConfig
# config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)

# # 3. Apply the quantization to your mlprogram model
# mlmodel_int8 = ct.optimize.coreml.linear_quantize_weights(mlmodel_int8, config=config)

# mlmodel_int8.author = "Chukwuebuka Emmanuel Igbokweuche"
# mlmodel_int8.short_description = "Rice disease classifier (INT8)"
# mlmodel_int8.version = "1.0"

# int8_path = os.path.join(OUTPUT_DIR, "RiceDiseaseClassifier_int8.mlpackage")
# mlmodel_int8.save(int8_path)
# int8_size = sum(f.stat().st_size for f in Path(int8_path).rglob('*') if f.is_file()) / (1024 * 1024)

# print(f"✓ Saved: {int8_size:.2f} MB")
# print(f"  Path: {int8_path}")

# # ============================================================
# # Summary
# # ============================================================
# print("\n" + "="*60)
# print("CONVERSION SUMMARY")
# print("="*60)
# print(f"{'Model':<30} {'Size (MB)':<15} {'Reduction':<15}")
# print("-"*60)
# print(f"{'FP32 Baseline':<30} {fp32_size:<15.2f} {'—':<15}")
# print(f"{'FP16 Optimized':<30} {fp16_size:<15.2f} {f'{(1-fp16_size/fp32_size)*100:.1f}%':<15}")
# print(f"{'INT8 Quantized':<30} {int8_size:<15.2f} {f'{(1-int8_size/fp32_size)*100:.1f}%':<15}")

# print("\n" + "="*60)
# print("DEPLOYMENT RECOMMENDATION")
# print("="*60)

# if int8_size < 1.0:
#     print("🏆 INT8 model is under 1MB!")
#     recommended = "RiceDiseaseClassifier_int8.mlpackage"
# elif fp16_size < 2.0:
#     print("✅ FP16 offers best balance")
#     recommended = "RiceDiseaseClassifier_fp16.mlpackage"
# else:
#     print("📱 Use FP32 for maximum accuracy")
#     recommended = "RiceDiseaseClassifier_fp32.mlpackage"

# print(f"\nRecommended: {recommended}")
# print(f"\n✅ All models support:")
# print(f"  • iOS Simulator & Devices")
# print(f"  • Neural Engine (when available)")
# print(f"  • CPU fallback")

# print(f"\n💡 To use in Xcode:")
# print(f"  1. Drag {recommended} into Xcode")
# print(f"  2. Xcode generates Swift interface")
# print(f"  3. Use: let model = try? RiceDiseaseClassifier_int8()")

# print(f"\n✅ Conversion complete!")

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
