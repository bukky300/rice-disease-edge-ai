package com.example.ricediseasedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Rice Disease Classifier using TFLite
 * Model: MobileNetV3Small with Dynamic Quantization
 * Input: 160x160 RGB image
 * Output: 6 disease classes
 */
class RiceClassifier(context: Context) {

    private var interpreter: Interpreter? = null

    // Class labels (must match training order)
    private val labels = listOf(
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf Scald",
        "Sheath Blight"
    )

    companion object {
        private const val MODEL_NAME = "rice_dynamic.tflite"  // 1.13 MB model
        private const val IMG_SIZE = 160
        private const val NUM_CLASSES = 6
        private const val NUM_THREADS = 4
    }

    init {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
                setUseXNNPACK(true)  // Enable hardware acceleration
            }
            val model = loadModelFile(context, MODEL_NAME)
            interpreter = Interpreter(model, options)
            android.util.Log.d("RiceClassifier", "Model loaded successfully")
        } catch (e: Exception) {
            android.util.Log.e("RiceClassifier", "Error loading model", e)
            throw RuntimeException("Failed to load TFLite model: ${e.message}")
        }
    }

    /**
     * Load TFLite model from assets
     */
    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Classify rice leaf image
     * @param bitmap Input image (any size, will be resized)
     * @return Pair of (predicted class name, confidence score)
     */
    fun classify(bitmap: Bitmap): ClassificationResult {

        val interpreter = this.interpreter
            ?: throw IllegalStateException("Interpreter not initialized")

        // Preprocess will handle resizing now
        val inputBuffer = preprocessImage(bitmap)

        // Run inference
        val outputArray = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter.run(inputBuffer, outputArray)

// 🔍 DEBUG: Log all probabilities
        val probabilities = outputArray[0]
        android.util.Log.d("RiceClassifier", "Raw probabilities:")
        probabilities.forEachIndexed { idx, prob ->
            android.util.Log.d("RiceClassifier", "  ${labels[idx]}: $prob")
        }

// Get predictions
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0 ?: 0
        val confidence = probabilities[maxIndex]

        return ClassificationResult(
            className = labels[maxIndex],
            confidence = confidence,
            allProbabilities = probabilities.mapIndexed { idx, prob ->
                labels[idx] to prob
            }.toMap()
        )
    }

    /**
     * Preprocess image using MobileNetV3 preprocessing
     * Converts image to float tensor with values in [-1, 1]
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Use FILTER_BILINEAR for better quality (closer to PIL's LANCZOS)
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            IMG_SIZE,
            IMG_SIZE,
            true  // Use bilinear filtering
        )

        val intValues = IntArray(IMG_SIZE * IMG_SIZE)
        resizedBitmap.getPixels(intValues, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        var pixel = 0
        for (i in 0 until IMG_SIZE) {
            for (j in 0 until IMG_SIZE) {
                val value = intValues[pixel++]

                val r = Color.red(value)
                val g = Color.green(value)
                val b = Color.blue(value)

                // MobileNetV3 preprocessing: must match training exactly
                inputBuffer.putFloat((r / 127.5f) - 1.0f)
                inputBuffer.putFloat((g / 127.5f) - 1.0f)
                inputBuffer.putFloat((b / 127.5f) - 1.0f)
            }
        }

        return inputBuffer
    }



    /**
     * Get model info for debugging
     */
    fun getModelInfo(): String {
        val interpreter = this.interpreter ?: return "Interpreter not initialized"

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        return """
            Model: $MODEL_NAME
            Input shape: ${inputTensor.shape().contentToString()}
            Input type: ${inputTensor.dataType()}
            Output shape: ${outputTensor.shape().contentToString()}
            Output type: ${outputTensor.dataType()}
            Num classes: $NUM_CLASSES
        """.trimIndent()
    }

    /**
     * Release resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Result of rice disease classification
 */
data class ClassificationResult(
    val className: String,
    val confidence: Float,
    val allProbabilities: Map<String, Float>
) {
    /**
     * Get confidence as percentage string
     */
    fun getConfidencePercent(): String = "%.1f%%".format(confidence * 100)

    /**
     * Check if prediction is confident enough
     */
    fun isConfident(threshold: Float = 0.5f): Boolean = confidence >= threshold

    /**
     * Get top N predictions
     */
    fun getTopPredictions(n: Int = 3): List<Pair<String, Float>> {
        return allProbabilities.entries
            .sortedByDescending { it.value }
            .take(n)
            .map { it.key to it.value }
    }
}