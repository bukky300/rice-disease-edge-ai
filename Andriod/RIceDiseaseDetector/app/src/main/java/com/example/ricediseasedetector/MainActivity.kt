package com.example.ricediseasedetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.ricediseasedetector.ui.theme.RiceDiseaseDetectorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private lateinit var classifier: RiceClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize classifier
        try {
            classifier = RiceClassifier(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        setContent {
            RiceDiseaseDetectorTheme {
                Surface(
                    modifier = Modifier.fillMaxSize()
                ) {
                    RiceClassifierScreen(classifier)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}

@Composable
fun RiceClassifierScreen(classifier: RiceClassifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var classificationResult by remember { mutableStateOf<ClassificationResult?>(null) }
    var isProcessing by remember { mutableStateOf(false) }
    var inferenceTime by remember { mutableStateOf(0L) }

    // Camera launcher
    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        bitmap?.let {
            selectedBitmap = it
            isProcessing = true
            classifyImage(it, classifier, scope) { result, time ->
                classificationResult = result
                inferenceTime = time
                isProcessing = false
            }
        }
    }

    // Gallery launcher
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                val bitmap = loadBitmapWithCorrectOrientation(context.contentResolver, it)
                selectedBitmap = bitmap
                isProcessing = true
                classifyImage(bitmap, classifier, scope) { result, time ->
                    classificationResult = result
                    inferenceTime = time
                    isProcessing = false
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Camera permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            cameraLauncher.launch(null)
        } else {
            Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title
        Text(
            text = "Rice Disease Classifier",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(vertical = 16.dp)
        )

        // Model Info Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors()
        ) {
            Text(
                text = classifier.getModelInfo(),
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(8.dp)
            )
        }

        // Image Preview
        Card(
            modifier = Modifier
                .size(300.dp)
                .padding(bottom = 16.dp)
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                selectedBitmap?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = "Selected rice leaf",
                        modifier = Modifier.fillMaxSize()
                    )
                } ?: Text("No image selected")
            }
        }

        // Action Buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    when (PackageManager.PERMISSION_GRANTED) {
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.CAMERA
                        ) -> {
                            cameraLauncher.launch(null)
                        }
                        else -> {
                            permissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    }
                },
                modifier = Modifier.weight(1f),
                enabled = !isProcessing
            ) {
                Text("📷 Camera")
            }

            Button(
                onClick = { galleryLauncher.launch("image/*") },
                modifier = Modifier.weight(1f),
                enabled = !isProcessing
            ) {
                Text("🖼️ Gallery")
            }
        }

        // Results
        if (isProcessing) {
            CircularProgressIndicator(modifier = Modifier.padding(32.dp))
            Text("Analyzing...")
        } else {
            classificationResult?.let { result ->
                ResultsCard(result, inferenceTime)
            }
        }

        // Instructions
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            colors = CardDefaults.cardColors()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Instructions",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = """
                        1. Take a photo or select from gallery
                        2. Make sure the rice leaf is clearly visible
                        3. Wait for analysis results
                        
                        Supported diseases:
                        • Bacterial Leaf Blight
                        • Brown Spot
                        • Healthy Rice Leaf
                        • Leaf Blast
                        • Leaf Scald
                        • Sheath Blight
                        
                        Note: Results may vary due to image quality
                    """.trimIndent(),
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
fun ResultsCard(result: ClassificationResult, inferenceTime: Long) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Results",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            // Main prediction
            Text(
                text = "Disease: ${result.className}",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(bottom = 4.dp)
            )

            Text(
                text = "Confidence: ${result.getConfidencePercent()}",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            // Warning for low confidence
            if (!result.isConfident(0.5f)) {
                Text(
                    text = "⚠️ Low confidence - results may be unreliable",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // Top 3 predictions
            Text(
                text = "Top 3 Predictions:",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp)
            )

            result.getTopPredictions(3).forEachIndexed { index, (label, prob) ->
                Text(
                    text = "${index + 1}. $label: ${"%.1f%%".format(prob * 100)}",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(vertical = 2.dp)
                )
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // Inference time
            Text(
                text = "Inference Time: ${inferenceTime}ms",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary
            )
        }
    }
}

private fun classifyImage(
    bitmap: Bitmap,
    classifier: RiceClassifier,
    scope: kotlinx.coroutines.CoroutineScope,
    onResult: (ClassificationResult, Long) -> Unit
) {
    scope.launch {
        val result = withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            val classification = classifier.classify(bitmap)
            val inferenceTime = System.currentTimeMillis() - startTime
            Pair(classification, inferenceTime)
        }
        onResult(result.first, result.second)
    }
}

private fun loadBitmapWithCorrectOrientation(
    contentResolver: android.content.ContentResolver,
    uri: Uri
): Bitmap {
    var bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

    // Fix orientation based on EXIF data
    contentResolver.openInputStream(uri)?.use { inputStream ->
        val exif = androidx.exifinterface.media.ExifInterface(inputStream)
        val orientation = exif.getAttributeInt(
            androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
            androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
        )

        bitmap = when (orientation) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90 ->
                rotateBitmap(bitmap, 90f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 ->
                rotateBitmap(bitmap, 180f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 ->
                rotateBitmap(bitmap, 270f)
            else -> bitmap
        }
    }

    return bitmap
}

private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
    val matrix = android.graphics.Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}