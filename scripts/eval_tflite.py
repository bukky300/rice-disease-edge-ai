import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Disable XNNPACK delegate to avoid runtime errors
os.environ['TF_ENABLE_XNNPACK'] = '0'

# CRITICAL: Load Keras preprocessing function
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models/tflite")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data/rice_split/test")

IMG_SIZE = (160, 160)
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
]

print("="*60)
print("TFLite Model Evaluation")
print("="*60)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Models Dir: {MODELS_DIR}")
print(f"Test Data: {TEST_DATA_DIR}")

# ============================================================
# Load Test Data
# ============================================================
def load_test_data(max_per_class=None):
    """Load test images and labels"""
    images = []
    labels = []
    
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"⚠️ Warning: {class_dir} not found")
            continue
        
        img_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
        img_paths.extend(glob.glob(os.path.join(class_dir, "*.png")))
        img_paths.extend(glob.glob(os.path.join(class_dir, "*.jpeg")))
        
        if max_per_class:
            img_paths = img_paths[:max_per_class]
        
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"  Loaded {len(img_paths)} images from {class_name}")
    
    return np.array(images), np.array(labels)

print("\nLoading test data...")
X_test, y_test = load_test_data()
print(f"✅ Loaded {len(X_test)} test samples\n")

# ============================================================
# Evaluation Function
# ============================================================
def evaluate_tflite_model(model_path, X_test, y_test, model_name):
    """Evaluate a TFLite model and return metrics"""
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return None
    
    print("="*60)
    print(f"Evaluating: {model_name}")
    print("="*60)
    
    # Load interpreter with CPU-only (no delegates)
    try:
        # Create interpreter without any delegates
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=4
        )
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Model info
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    print(f"Input dtype: {input_details['dtype']}")
    print(f"Input shape: {input_details['shape']}")
    print(f"Output dtype: {output_details['dtype']}")
    
    # Run inference on all test samples
    predictions = []
    inference_times = []
    
    for i, img in enumerate(X_test):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(X_test)}...", end='\r')
        
        try:
            # Preprocess EXACTLY as in training using Keras function
            input_data = img.astype(np.float32)
            # Use the same preprocessing as training
            input_data = preprocess_input(input_data)
            input_data = input_data[None, ...]
            
            # Inference
            start = time.perf_counter()
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            end = time.perf_counter()
            
            output = interpreter.get_tensor(output_details['index'])[0]
            
            # Handle quantized output if needed
            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output.astype(np.float32) - zero_point)
            
            pred = np.argmax(output)
            predictions.append(pred)
            inference_times.append((end - start) * 1000)  # Convert to ms
        except Exception as e:
            print(f"\n❌ Error on sample {i}: {e}")
            predictions.append(0)  # Default prediction
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
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))
    
    return {
        "model_name": model_name,
        "file_size_mb": file_size,
        "accuracy": accuracy * 100,
        "avg_inference_ms": avg_time,
        "std_inference_ms": std_time,
        "predictions": predictions
    }

# ============================================================
# Evaluate All Models
# ============================================================
models_to_eval = [
    ("rice_fp32.tflite", "FP32 Baseline"),
    ("rice_dynamic.tflite", "Dynamic Quantized"),
    ("rice_int8.tflite", "INT8 Quantized"),
]

results = []

for model_file, model_name in models_to_eval:
    model_path = os.path.join(MODELS_DIR, model_file)
    result = evaluate_tflite_model(model_path, X_test, y_test, model_name)
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
    print(f"{'Model':<25} {'Size (MB)':<15} {'Accuracy (%)':<15} {'Inference (ms)':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model_name']:<25} {r['file_size_mb']:<15.2f} {r['accuracy']:<15.2f} {r['avg_inference_ms']:<15.2f}")
    
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        baseline = results[0]
        best_compressed = results[-1]
        
        size_reduction = (1 - best_compressed['file_size_mb'] / baseline['file_size_mb']) * 100
        accuracy_drop = baseline['accuracy'] - best_compressed['accuracy']
        speedup = baseline['avg_inference_ms'] / best_compressed['avg_inference_ms']
        
        print(f"Size Reduction: {size_reduction:.1f}%")
        print(f"Accuracy Drop: {accuracy_drop:.2f}%")
        print(f"Speedup: {speedup:.2f}x")
        
        if best_compressed['file_size_mb'] < 1.0:
            print("\n🏆 BONUS ACHIEVED! Model is under 1MB")
        elif best_compressed['file_size_mb'] < 5.0:
            print("\n✅ Model meets <5MB requirement")
        
    print("\n💡 Hardware Specs:")
    print("   CPU: [Specify your CPU model]")
    print("   RAM: [Specify your RAM]")
    print("   OS: [Specify your OS]")
    
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
        output_path = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "confusion_matrix_int8.png"), dpi=150)
        print(f"\n✅ Confusion matrix saved to: {output_path}/confusion_matrix_int8.png")
        plt.show()
        
    # ============================================================
    # Size Comparison Chart
    # ============================================================
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        model_names = [r['model_name'] for r in results]
        sizes = [r['file_size_mb'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        # Size comparison
        ax1.bar(model_names, sizes, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_ylabel('Size (MB)')
        ax1.set_title('Model Size Comparison')
        ax1.axhline(y=5.0, color='r', linestyle='--', label='5MB Limit')
        ax1.axhline(y=1.0, color='g', linestyle='--', label='1MB Bonus')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Accuracy comparison
        ax2.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Model Accuracy Comparison')
        ax2.set_ylim([min(accuracies) - 5, 100])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "model_comparison.png"), dpi=150)
        print(f"✅ Comparison chart saved to: {output_path}/model_comparison.png")
        plt.show()

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)