import SwiftUI

struct ContentView: View {
    @State private var classifier = RiceClassifier()
    @State private var selectedImage: UIImage?
    @State private var showingImagePicker = false
    @State private var showingCamera = false
    @State private var sourceType: UIImagePickerController.SourceType = .photoLibrary
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Title
                    Text("Rice Disease Detector")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .padding(.top)
                    
                    // Model Info Card
                    GroupBox {
                        Text("Model: MobileNetV3-Small INT8")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("Size: 1.24 MB | 6 Classes")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                    
                    // Image Preview
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 300)
                            .cornerRadius(12)
                            .shadow(radius: 4)
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.2))
                            .frame(height: 300)
                            .overlay(
                                Text("No image selected")
                                    .foregroundColor(.secondary)
                            )
                    }
                    
                    // Action Buttons
                    HStack(spacing: 16) {
                        Button(action: {
                            sourceType = .camera
                            showingCamera = true
                        }) {
                            Label("Camera", systemImage: "camera")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        
                        Button(action: {
                            sourceType = .photoLibrary
                            showingImagePicker = true
                        }) {
                            Label("Gallery", systemImage: "photo")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .padding(.horizontal)
                    
                    // Results Card
                    if classifier.isAnalyzing {
                        ProgressView("Analyzing...")
                            .padding()
                    } else if let result = classifier.classificationResult {
                        ResultsView(result: result, inferenceTime: classifier.inferenceTime)
                    }
                    
                    // Instructions
                    GroupBox(label: Text("Instructions").fontWeight(.semibold)) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("1. Take a photo or select from gallery")
                            Text("2. Ensure rice leaf is clearly visible")
                            Text("3. Wait for analysis results")
                            Text("\nSupported diseases:")
                                .fontWeight(.semibold)
                                .padding(.top, 4)
                            ForEach(classifier.classLabels, id: \.self) { label in
                                Text("• \(label)")
                                    .font(.caption)
                            }
                        }
                        .font(.subheadline)
                    }
                    .padding(.horizontal)
                    .padding(.bottom)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(image: $selectedImage, sourceType: sourceType)
                .onDisappear {
                    if let image = selectedImage {
                        classifier.classify(image: image)
                    }
                }
        }
        .fullScreenCover(isPresented: $showingCamera) {
            ImagePicker(image: $selectedImage, sourceType: .camera)
                .onDisappear {
                    if let image = selectedImage {
                        classifier.classify(image: image)
                    }
                }
        }
    }
}

struct ResultsView: View {
    let result: ClassificationResult
    let inferenceTime: Double
    
    var body: some View {
        GroupBox(label: Text("Results").fontWeight(.semibold)) {
            VStack(alignment: .leading, spacing: 12) {
                // Main prediction
                HStack {
                    Text("Disease:")
                        .fontWeight(.semibold)
                    Spacer()
                    Text(result.className)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                }
                
                HStack {
                    Text("Confidence:")
                        .fontWeight(.semibold)
                    Spacer()
                    Text(String(format: "%.1f%%", result.confidence * 100))
                        .fontWeight(.bold)
                        .foregroundColor(result.confidence > 0.5 ? .green : .orange)
                }
                
                if result.confidence < 0.5 {
                    Text("⚠️ Low confidence - results may be unreliable")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
                
                Divider()
                
                // Top 3 predictions
                Text("Top 3 Predictions:")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                ForEach(Array(result.topPredictions.prefix(3).enumerated()), id: \.offset) { index, prediction in
                    HStack {
                        Text("\(index + 1). \(prediction.label)")
                            .font(.caption)
                        Spacer()
                        Text(String(format: "%.1f%%", prediction.probability * 100))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Divider()
                
                // Inference time
                Text("Inference Time: \(String(format: "%.0f", inferenceTime))ms")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
