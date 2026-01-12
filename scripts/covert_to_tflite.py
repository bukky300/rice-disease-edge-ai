import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path

# ============================================================
# Configuration - UPDATE THESE PATHS
# ============================================================
# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# If running from scripts folder, go up one level
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

print(f"Project Root: {PROJECT_ROOT}")

# Input paths (absolute)
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/baseline/mobilenetv3small")
VAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/rice_split/val")

# Output paths (absolute)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/tflite")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFLITE_FP32_PATH = os.path.join(OUTPUT_DIR, "rice_fp32.tflite")
TFLITE_DYNAMIC_PATH = os.path.join(OUTPUT_DIR, "rice_dynamic.tflite")
TFLITE_INT8_PATH = os.path.join(OUTPUT_DIR, "rice_int8.tflite")

IMG_SIZE = (160, 160)

# Verify paths exist
print("\nVerifying paths...")
if not os.path.exists(SAVED_MODEL_DIR):
    raise FileNotFoundError(f"❌ SavedModel not found: {SAVED_MODEL_DIR}")
if not os.path.exists(VAL_DATA_DIR):
    raise FileNotFoundError(f"❌ Validation data not found: {VAL_DATA_DIR}")

print(f"✓ SavedModel found: {SAVED_MODEL_DIR}")
print(f"✓ Validation data found: {VAL_DATA_DIR}")
print(f"✓ Output directory: {OUTPUT_DIR}")

# ============================================================
# Representative Dataset Generator (for INT8 quantization)
# ============================================================
def representative_dataset_gen():
    """Generate representative samples for quantization calibration"""
    image_paths = []
    
    # Collect sample images from each class
    for class_name in os.listdir(VAL_DATA_DIR):
        class_path = os.path.join(VAL_DATA_DIR, class_name)
        if os.path.isdir(class_path):
            imgs = glob.glob(os.path.join(class_path, "*.jpg"))[:15]  # 15 per class
            image_paths.extend(imgs)
    
    print(f"  Using {len(image_paths)} calibration samples")
    
    for img_path in image_paths:
        # Load and preprocess exactly as in training
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        
        # Apply MobileNetV3 preprocessing: scale to [-1, 1]
        arr = (arr / 127.5) - 1.0
        
        yield [arr[None, ...]]

# ============================================================
# 1. FP32 Model (Baseline - No Quantization)
# ============================================================
print("\n" + "="*60)
print("1. Converting FP32 Model (Baseline)")
print("="*60)

converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_fp32.optimizations = []  # No optimization
converter_fp32.target_spec.supported_types = [tf.float32]

tflite_fp32 = converter_fp32.convert()

with open(TFLITE_FP32_PATH, "wb") as f:
    f.write(tflite_fp32)

size_fp32 = os.path.getsize(TFLITE_FP32_PATH) / (1024 * 1024)
print(f"✅ FP32 Model saved")
print(f"   Path: {TFLITE_FP32_PATH}")
print(f"   Size: {size_fp32:.2f} MB")

# ============================================================
# 2. Dynamic Range Quantization (Weights to INT8)
# ============================================================
print("\n" + "="*60)
print("2. Converting with Dynamic Range Quantization")
print("="*60)

converter_dynamic = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_dynamic = converter_dynamic.convert()

with open(TFLITE_DYNAMIC_PATH, "wb") as f:
    f.write(tflite_dynamic)

size_dynamic = os.path.getsize(TFLITE_DYNAMIC_PATH) / (1024 * 1024)
print(f"✅ Dynamic Quantized Model saved")
print(f"   Path: {TFLITE_DYNAMIC_PATH}")
print(f"   Size: {size_dynamic:.2f} MB")
print(f"   Reduction: {(1 - size_dynamic/size_fp32)*100:.1f}%")

# ============================================================
# 3. Full INT8 Quantization (Weights + Activations)
# ============================================================
print("\n" + "="*60)
print("3. Converting with Full INT8 Quantization")
print("="*60)

converter_int8 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen

# Allow fallback to ensure compatibility (fixes XNNPACK issues)
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback to FP32 for incompatible ops
]
converter_int8.inference_input_type = tf.float32
converter_int8.inference_output_type = tf.float32

tflite_int8 = converter_int8.convert()

with open(TFLITE_INT8_PATH, "wb") as f:
    f.write(tflite_int8)

size_int8 = os.path.getsize(TFLITE_INT8_PATH) / (1024 * 1024)
print(f"✅ INT8 Quantized Model saved")
print(f"   Path: {TFLITE_INT8_PATH}")
print(f"   Size: {size_int8:.2f} MB")
print(f"   Reduction: {(1 - size_int8/size_fp32)*100:.1f}%")

# ============================================================
# Summary Table
# ============================================================
print("\n" + "="*60)
print("CONVERSION SUMMARY")
print("="*60)
print(f"{'Model Type':<25} {'Size (MB)':<15} {'Reduction':<15}")
print("-"*60)
print(f"{'FP32 Baseline':<25} {size_fp32:<15.2f} {'---':<15}")
print(f"{'Dynamic Quantized':<25} {size_dynamic:<15.2f} {f'{(1-size_dynamic/size_fp32)*100:.1f}%':<15}")
print(f"{'INT8 Quantized':<25} {size_int8:<15.2f} {f'{(1-size_int8/size_fp32)*100:.1f}%':<15}")

print("\n" + "="*60)
print("DEPLOYMENT RECOMMENDATIONS")
print("="*60)
if size_int8 < 1.0:
    print("🏆 BONUS ACHIEVED! Model is under 1MB")
    print(f"   Use: {os.path.basename(TFLITE_INT8_PATH)} for edge deployment")
elif size_int8 < 5.0:
    print(f"✅ Model meets <5MB requirement")
    print(f"   Use: {os.path.basename(TFLITE_INT8_PATH)} for edge deployment")
else:
    print(f"⚠️  Model is {size_int8:.2f}MB (target: <5MB)")
    print(f"   Consider pruning or using MobileNetV3-Nano")

print("\n💡 Next step: Run evaluation script to check accuracy")
print(f"💡 All models saved to: {OUTPUT_DIR}")