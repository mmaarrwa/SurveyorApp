import Foundation
import ARKit
import SceneKit
import simd
import Vision
import CoreML

final class ARManager: NSObject, ObservableObject, ARSessionDelegate {
    static let shared = ARManager()

    let sceneView: ARSCNView = {
        let v = ARSCNView(frame: .zero)
        v.autoenablesDefaultLighting = true
        v.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
        return v
    }()

    @Published var isStreaming: Bool = false
    @Published var statusText: String = "Ready. Enter IP & Connect."
    @Published var serverIP: String = "192.168.1.10"
    
    @Published var cameraHeightInput: String = "0.20"
    
    var robotCameraHeight: Float {
        return Float(cameraHeightInput) ?? 0.20
    }

    private let network = NetworkManager.shared

    private let rayGridSize = 5            
    private let rayScreenRadius: CGFloat = 0.15 
    private let maxRayDistance: Float = 3.0
    private let minRayDistance: Float = 0.15
    private let featurePointConeHalfWidth: Float = 0.25
    private let featurePointConeHalfHeight: Float = 0.25
    private let featurePointNearZ: Float = -0.2
    private let featurePointFarZ: Float = -2.5
    private let featurePointDensityThreshold = 60
    private let densityFallbackMinDistance: Float = 0.4

    private var smoothedObstacleDist: Float = 10.0 
    private let smoothingAlpha: Float = 0.2 

    private var visionModel: VNCoreMLModel?
    private var visionRequest: VNCoreMLRequest?
    private var isProcessingFrame = false

    override init() {
        super.init()
        // THE FIX: Ensure the camera feed reaches our code!
        sceneView.session.delegate = self 
        
        setupYOLO()
        network.onCommandReceived = { [weak self] command in
            self?.handleRemoteCommand(command)
        }
    }

    private func setupYOLO() {
        guard let modelURL = Bundle.main.url(forResource: "yolo26n", withExtension: "mlmodelc") else {
            print("❌ YOLO model not found in app bundle!")
            return
        }
        
        do {
            let coreMLModel = try MLModel(contentsOf: modelURL)
            visionModel = try VNCoreMLModel(for: coreMLModel)
            visionRequest = VNCoreMLRequest(model: visionModel!) { request, error in }
            visionRequest?.imageCropAndScaleOption = .scaleFill
            print("✅ YOLO Model Loaded Successfully!")
        } catch {
            print("❌ Failed to load Vision ML model: \(error)")
        }
    }

    func startSessionIfNeeded() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity 
        sceneView.session.run(config)
    }

    func connectToNetwork() {
        network.start(ipAddress: serverIP)
        statusText = "Connected. Waiting for START..."
    }

    func handleRemoteCommand(_ command: String) {
        let cleanCommand = command.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        
        if cleanCommand == "START" && !isStreaming {
            DispatchQueue.main.async { self.toggleStreaming() }
        } else if cleanCommand == "STOP" && isStreaming {
            DispatchQueue.main.async { self.toggleStreaming() }
        }
    }

    func toggleStreaming() {
        isStreaming.toggle()

        if isStreaming {
            statusText = "Streaming & Detecting..."
            smoothedObstacleDist = 10.0 
            
            let config = ARWorldTrackingConfiguration()
            config.worldAlignment = .gravity
            sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        } else {
            statusText = "Stopped"
            sceneView.session.pause()
        }
    }

    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isStreaming else { return }

        let cameraTransform = frame.camera.transform
        let col3 = cameraTransform.columns.3
        let currentPos = SIMD3<Float>(col3.x, col3.y, col3.z)
        let q = simd_quatf(cameraTransform)

        let obstacleDistance = detectObstacleDistance(frame: frame)

        let navPacket: [String: Any] = [
            "type": "nav",
            "timestamp": frame.timestamp,
            "position": [currentPos.x, currentPos.y, currentPos.z],
            "orientation": [q.vector.x, q.vector.y, q.vector.z, q.vector.w],
            "obstacle_dist": obstacleDistance
        ]
        network.sendPose(navPacket)

        if !isProcessingFrame, let request = visionRequest {
            isProcessingFrame = true
            let pixelBuffer = frame.capturedImage
            
            DispatchQueue.global(qos: .userInitiated).async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
                do {
                    try handler.perform([request])
                    self.processYOLOResults(request: request, frame: frame, currentPos: currentPos)
                } catch {
                    print("❌ YOLO Execution Error: \(error)")
                    self.isProcessingFrame = false
                }
            }
        }
    }

    // MARK: - Process AI & 3D Math
    // MARK: - Process AI & 3D Math (UNFILTERED FIREHOSE MODE)
    private func processYOLOResults(request: VNRequest, frame: ARFrame, currentPos: SIMD3<Float>) {
        defer { isProcessingFrame = false }
        
        guard let results = request.results as? [VNRecognizedObjectObservation] else { 
            network.sendLog("⚠️ YOLO ERROR: Output is NOT Bounding Boxes! Check Colab export.")
            return 
        }
        
        // This loop automatically handles MULTIPLE objects detected in the same frame!
        for observation in results {
            guard let topLabel = observation.labels.first else { continue }
            
            // 🚨 WE COMPLETELY DELETED THE CONFIDENCE FILTER! 🚨
            // It will now process EVERYTHING it sees, even if it is only 1% confident.
            
            let confPct = Int(topLabel.confidence * 100)
            network.sendLog("👀 YOLO sees: \(topLabel.identifier) (\(confPct)%)")
            
            let bbox = observation.boundingBox
            let arKitCenter = CGPoint(x: bbox.midX, y: 1.0 - bbox.minY) 
            var finalObjPos: SIMD3<Float>? = nil
            
            let hitTestResults = frame.hitTest(arKitCenter, types: [.featurePoint, .estimatedHorizontalPlane])
            
            if let hit = hitTestResults.first {
                finalObjPos = SIMD3<Float>(hit.worldTransform.columns.3.x,
                                           hit.worldTransform.columns.3.y,
                                           hit.worldTransform.columns.3.z)
            } else {
                let intrinsics = frame.camera.intrinsics
                let fx = intrinsics[0][0]; let fy = intrinsics[1][1]
                let cx = intrinsics[2][0]; let cy = intrinsics[2][1]
                
                let imageRes = frame.camera.imageResolution
                let ub = Float(arKitCenter.x * imageRes.width)
                let vb = Float(arKitCenter.y * imageRes.height)
                
                let xn = (ub - cx) / fx
                let yn = (vb - cy) / fy
                let rc = simd_float3(xn, yn, 1.0)
                
                let cameraTransform = frame.camera.transform
                let R = simd_float3x3(
                    simd_float3(cameraTransform.columns.0.x, cameraTransform.columns.0.y, cameraTransform.columns.0.z),
                    simd_float3(cameraTransform.columns.1.x, cameraTransform.columns.1.y, cameraTransform.columns.1.z),
                    simd_float3(cameraTransform.columns.2.x, cameraTransform.columns.2.y, cameraTransform.columns.2.z)
                )
                
                let rw = R * rc
                
                if rw.y < -0.001 {
                    let t = -robotCameraHeight / rw.y
                    let objX = currentPos.x + (t * rw.x)
                    let objZ = currentPos.z + (t * rw.z)
                    let objY = currentPos.y - robotCameraHeight 
                    finalObjPos = SIMD3<Float>(objX, objY, objZ)
                } else {
                    network.sendLog("⚠️ MATH FAIL: Ray pointing up for \(topLabel.identifier)")
                }
            }
            
            if let objPos = finalObjPos {
                let distToObject = distance(currentPos, objPos)
                let formattedDist = String(format: "%.2f", distToObject)
                
                // Add the confidence % to the floating 3D text!
                self.add3DLabel(text: "\(topLabel.identifier): \(formattedDist)m (\(confPct)%)", position: objPos)
                
                let detectionPacket: [String: Any] = [
                    "type": "survey",
                    "timestamp": frame.timestamp,
                    "position": [objPos.x, objPos.y, objPos.z],
                    "label": topLabel.identifier,
                    // Add the confidence % to the Python graph!
                    "note": "Dist: \(formattedDist)m, Conf: \(confPct)%"
                ]
                network.sendPose(detectionPacket)
            }
        }
    }

    // --- NEW: 3D AR LABEL FUNCTION ---
    private func add3DLabel(text: String, position: SIMD3<Float>) {
        DispatchQueue.main.async {
            // Create 3D text
            let textGeometry = SCNText(string: text, extrusionDepth: 0.01)
            textGeometry.font = UIFont.systemFont(ofSize: 1.0)
            textGeometry.firstMaterial?.diffuse.contents = UIColor.green
            
            let textNode = SCNNode(geometry: textGeometry)
            
            // Scale it down to look normal in AR (5% of original size)
            textNode.scale = SCNVector3(0.05, 0.05, 0.05)
            
            // Position it 20cm above the object's hit point
            textNode.position = SCNVector3(position.x, position.y + 0.20, position.z)
            
            // Force the text to always face the camera
            let billboardConstraint = SCNBillboardConstraint()
            billboardConstraint.freeAxes = .Y
            textNode.constraints = [billboardConstraint]
            
            self.sceneView.scene.rootNode.addChildNode(textNode)
            
            // Fade out and remove after 2 seconds to avoid screen clutter
            let fadeOut = SCNAction.fadeOut(duration: 2.0)
            let remove = SCNAction.removeFromParentNode()
            textNode.runAction(SCNAction.sequence([fadeOut, remove]))
        }
    }

    // MARK: - General Obstacle detection helpers
    private func detectObstacleDistance(frame: ARFrame) -> Float {
        var rawDistance: Float = 10.0 
        if let hitDist = performGridHitTest(cameraTransform: frame.camera.transform) {
            rawDistance = hitDist
        } else if let densityDist = featurePointDensityFallback(frame: frame) {
            rawDistance = densityDist
        }
        smoothedObstacleDist = (smoothingAlpha * rawDistance) + ((1.0 - smoothingAlpha) * smoothedObstacleDist)
        return smoothedObstacleDist
    }

    private func performGridHitTest(cameraTransform: simd_float4x4) -> Float? {
        let view = sceneView
        let bounds = view.bounds
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        let minSide = min(bounds.width, bounds.height)
        let radiusPx = rayScreenRadius * minSide
        let camPos = SIMD3<Float>(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
        
        var nearestDistance: Float? = nil
        var hitCount = 0
        let requiredHits = 2 

        let half = (rayGridSize - 1) / 2
        for i in 0..<rayGridSize {
            for j in 0..<rayGridSize {
                let nx = CGFloat(i - half) / CGFloat(max(1, half))
                let ny = CGFloat(j - half) / CGFloat(max(1, half))
                let samplePoint = CGPoint(x: center.x + nx * radiusPx, y: center.y + ny * radiusPx)
                let results = view.hitTest(samplePoint, types: [.featurePoint, .existingPlaneUsingExtent, .estimatedHorizontalPlane])
                
                if let hit = results.first {
                    let hitPos = SIMD3<Float>(hit.worldTransform.columns.3.x, hit.worldTransform.columns.3.y, hit.worldTransform.columns.3.z)
                    let d = distance(camPos, hitPos)
                    if d >= minRayDistance && d <= maxRayDistance {
                        hitCount += 1
                        if let current = nearestDistance { nearestDistance = min(current, d) } else { nearestDistance = d }
                    }
                }
            }
        }
        if hitCount >= requiredHits { return nearestDistance }
        return nil
    }

    private func featurePointDensityFallback(frame: ARFrame) -> Float? {
        guard let points = frame.rawFeaturePoints?.points else { return nil }
        let cameraTransform = frame.camera.transform
        let worldToCamera = cameraTransform.inverse
        var count = 0
        var nearestZ: Float? = nil

        for p in points {
            let worldPoint = simd_float4(p.x, p.y, p.z, 1)
            let local = simd_mul(worldToCamera, worldPoint)

            if local.z < featurePointNearZ && local.z > featurePointFarZ {
                if abs(local.x) <= featurePointConeHalfWidth && abs(local.y) <= featurePointConeHalfHeight {
                    count += 1
                    if nearestZ == nil || abs(local.z) < nearestZ! { nearestZ = abs(local.z) }
                }
            }
        }
        if count >= featurePointDensityThreshold {
            if let nz = nearestZ { return max(minRayDistance, min(maxRayDistance, nz)) } else { return densityFallbackMinDistance }
        } else { return nil }
    }
}