import Foundation
import CoreML
import Vision
import UIKit
import Observation

struct ClassificationResult {
    let className: String
    let confidence: Float
    let topPredictions: [(label: String, probability: Float)]
}

@Observable
class RiceClassifier {
    var classificationResult: ClassificationResult?
    var isAnalyzing = false
    var inferenceTime: Double = 0
    
    private var model: VNCoreMLModel?
    
    let classLabels = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf Scald",
        "Sheath Blight"
    ]
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            // Load the CoreML model
            let config = MLModelConfiguration()
            config.computeUnits = .all // Use Neural Engine + GPU + CPU
            
            let coreMLModel = try RiceDiseaseClassifier_int8(configuration: config)
            model = try VNCoreMLModel(for: coreMLModel.model)
            
            print("✅ Model loaded successfully")
        } catch {
            print("❌ Failed to load model: \(error.localizedDescription)")
        }
    }
    
    func classify(image: UIImage) {
        guard let model = model else {
            print("❌ Model not loaded")
            return
        }
        
        guard let ciImage = CIImage(image: image) else {
            print("❌ Failed to convert UIImage to CIImage")
            return
        }
        
        isAnalyzing = true
        classificationResult = nil
        
        // Create Vision request
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            
            if let error = error {
                print("❌ Classification error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.isAnalyzing = false
                }
                return
            }
            
            self.processResults(request.results)
        }
        
        // Configure request
        request.imageCropAndScaleOption = .centerCrop
        
        // Perform request
        let startTime = Date()
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
                let endTime = Date()
                let inferenceTime = endTime.timeIntervalSince(startTime) * 1000 // Convert to ms
                
                DispatchQueue.main.async {
                    self.inferenceTime = inferenceTime
                }
            } catch {
                print("❌ Failed to perform request: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.isAnalyzing = false
                }
            }
        }
    }
    
    private func processResults(_ results: [Any]?) {
        guard let results = results as? [VNClassificationObservation],
              !results.isEmpty else {
            print("❌ No results")
            DispatchQueue.main.async {
                self.isAnalyzing = false
            }
            return
        }
        
        // Get top prediction
        let topResult = results[0]
        
        // Get all predictions sorted by confidence
        let topPredictions = results.map { observation in
            (label: observation.identifier, probability: observation.confidence)
        }
        
        let result = ClassificationResult(
            className: topResult.identifier,
            confidence: topResult.confidence,
            topPredictions: topPredictions
        )
        
        DispatchQueue.main.async {
            self.classificationResult = result
            self.isAnalyzing = false
            
            print("✅ Classification complete:")
            print("   Disease: \(result.className)")
            print("   Confidence: \(String(format: "%.1f%%", result.confidence * 100))")
        }
    }
}
