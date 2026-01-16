#!/usr/bin/env python3
"""
Evaluate CoreML models on test set
"""

from pyexpat import model
from unittest import result
import coremltools as ct
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models/coreml"
TEST_DATA_DIR = PROJECT_ROOT / "data/rice_split/test"

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

IMG_SIZE = 160

print("="*60)
print("CoreML Model Evaluation")
print("="*60)
print(f"Models: {MODELS_DIR}")
print(f"Test Data: {TEST_DATA_DIR}")

# ============================================================
# Load Test Data
# ============================================================
def load_test_data(max_per_class=None):
    """Load test images and labels"""
    images = []
    labels = []
    paths = []
    
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = TEST_DATA_DIR / class_name
        if not class_dir.is_dir():
            print(f"⚠️ Warning: {class_dir} not found")
            continue
        
        img_paths = list(class_dir.glob("*.jpg"))
        img_paths.extend(class_dir.glob("*.png"))
        img_paths.extend(class_dir.glob("*.jpeg"))
        
        if max_per_class:
            img_paths = img_paths[:max_per_class]
        
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(label_idx)
                paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"  Loaded {len(img_paths)} images from {class_name}")
    
    return images, np.array(labels), paths

print("\nLoading test data...")
X_test, y_test, test_paths = load_test_data()
print(f"✅ Loaded {len(X_test)} test samples\n")

# ============================================================
# Evaluate CoreML Model
# ============================================================
def evaluate_coreml_model(model_path, X_test, y_test):
    """Evaluate a CoreML model"""
    
    if not model_path.exists():
        print(f"⚠️ Model not found: {model_path}")
        return None
    
    print("="*60)
    print(f"Evaluating: {model_path.name}")
    print("="*60)
    
    # Load model
    try:
        model = ct.models.MLModel(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Get model size
    model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    
    # Get model spec
    spec = model.get_spec()
    print(f"Model type: {spec.WhichOneof('Type')}")
    
    # Run inference
    predictions = []
    inference_times = []
    
    for i, img in enumerate(X_test):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(X_test)}...", end='\r')
        
        try:
            # Apply MobileNetV3 preprocessing manually
            img_float = img.astype(np.float32)
            img_preprocessed = (img_float / 127.5) - 1.0
            
            # Create input dictionary (model expects 'inputs' key)

            input_name = list(model.get_spec().description.input)[0].name

            input_dict = {
                input_name: img_preprocessed[np.newaxis, :, :, :]
            }
            
            # Run prediction
            start = time.perf_counter()
            result = model.predict(input_dict)
            result = model.predict(input_dict)
            end = time.perf_counter()
            
            # Get prediction from classLabel
            # Get output names from spec
            output_names = [o.name for o in model.get_spec().description.output]

            if "classLabel" in output_names:
                pred_label = result["classLabel"]
                pred_idx = CLASS_NAMES.index(pred_label)

            elif "classLabelProbs" in output_names:
                probs = result["classLabelProbs"]
                pred_label = max(probs, key=probs.get)
                pred_idx = CLASS_NAMES.index(pred_label)

            else:
                # Raw probability / logits tensor
                output_name = output_names[0]
                vec = result[output_name][0]
                pred_idx = int(np.argmax(vec))
            
            predictions.append(pred_idx)
            inference_times.append((end - start) * 1000)  # ms
            
        except Exception as e:
            print(f"\n❌ Error on sample {i}: {e}")
            predictions.append(0)
            inference_times.append(0)
    
    print(f"  Processing {len(X_test)}/{len(X_test)}... Done!")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-"*40)
    print(f"{'Accuracy':<25} {accuracy*100:.2f}%")
    print(f"{'Avg Inference Time':<25} {avg_time:.2f} ms")
    print(f"{'Std Inference Time':<25} {std_time:.2f} ms")
    print(f"{'Min Inference Time':<25} {np.min(inference_times):.2f} ms")
    print(f"{'Max Inference Time':<25} {np.max(inference_times):.2f} ms")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, zero_division=0))
    
    return {
        "model_name": model_path.stem,
        "file_size_mb": model_size,
        "accuracy": accuracy * 100,
        "avg_inference_ms": avg_time,
        "std_inference_ms": std_time,
        "predictions": predictions
    }

# ============================================================
# Evaluate All Models
# ============================================================
models_to_eval = [
    "RiceDiseaseClassifier_fp32.mlpackage",
    "RiceDiseaseClassifier_fp16.mlpackage",
    "RiceDiseaseClassifier_int8.mlpackage",
]

results = []

for model_name in models_to_eval:
    model_path = MODELS_DIR / model_name
    result = evaluate_coreml_model(model_path, X_test, y_test)
    if result:
        results.append(result)
    print("\n")

# ============================================================
# Summary Table
# ============================================================
if results:
    print("="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Size (MB)':<15} {'Accuracy (%)':<15} {'Inference (ms)':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model_name']:<30} {r['file_size_mb']:<15.2f} {r['accuracy']:<15.2f} {r['avg_inference_ms']:<15.2f}")
    
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        baseline = results[0]
        best_compressed = results[-1]
        
        size_reduction = (1 - best_compressed['file_size_mb'] / baseline['file_size_mb']) * 100
        accuracy_drop = baseline['accuracy'] - best_compressed['accuracy']
        
        print(f"Size Reduction: {size_reduction:.1f}%")
        print(f"Accuracy Drop: {accuracy_drop:.2f}%")
        
        if best_compressed['file_size_mb'] < 1.0:
            print("\n🏆 INT8 model is under 1MB!")
        elif best_compressed['file_size_mb'] < 2.0:
            print("\n✅ Model meets size requirements")
    
    print("\n💡 Hardware: macOS (CoreML)")
    print("💡 Dataset: Rice Disease Test Set")
    
    # ============================================================
    # Confusion Matrix for Best Model
    # ============================================================
    if results:
        best_model = results[-1]  # INT8 model
        cm = confusion_matrix(y_test, best_model['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix - {best_model["model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        output_path = PROJECT_ROOT / "results"
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "confusion_matrix_coreml.png", dpi=150)
        print(f"\n✅ Confusion matrix saved to: {output_path}/confusion_matrix_coreml.png")
        plt.show()

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)

