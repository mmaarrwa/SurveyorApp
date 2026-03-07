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
        // Turn off feature points visually so the screen is cleaner
        // v.debugOptions = [ARSCNDebugOptions.showFeaturePoints] 
        return v
    }()

    @Published var isStreaming: Bool = false
    @Published var statusText: String = "Ready. Enter IP & Connect."
    @Published var serverIP: String = "192.168.1.10"
    @Published var cameraHeightInput: String = "0.20"
    
    var robotCameraHeight: Float { return Float(cameraHeightInput) ?? 0.20 }

    private let network = NetworkManager.shared

    // --- Smoothing & Confidence ---
    private var smoothedObstacleDist: Float = 10.0 
    private let smoothingAlpha: Float = 0.2 

    private var visionModel: VNCoreMLModel?
    private var visionRequest: VNCoreMLRequest?
    private var isProcessingFrame = false

    override init() {
        super.init()
        // 🚨 CRITICAL: This ensures ARKit actually sends frames to our code!
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
            
            // 🚀 ENABLE LiDAR HARDWARE FEATURES (Automatically skipped on older iPhones) 🚀
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                config.frameSemantics.insert(.smoothedSceneDepth)
                statusText = "LiDAR Streaming..."
            } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                config.frameSemantics.insert(.sceneDepth)
                statusText = "LiDAR Streaming..."
            }
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                config.sceneReconstruction = .mesh
            }
            
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
                    self.isProcessingFrame = false
                }
            }
        }
    }

    // MARK: - Process AI & Hybrid 3D Math
    private func processYOLOResults(request: VNRequest, frame: ARFrame, currentPos: SIMD3<Float>) {
        defer { isProcessingFrame = false }
        
        // 🚨 THE RESTORED SAFETY NET 🚨
        guard let results = request.results as? [VNRecognizedObjectObservation] else { 
            network.sendLog("⚠️ YOLO ERROR: Output is NOT Bounding Boxes! Check Colab export.")
            return 
        }
        
        for observation in results {
            guard let topLabel = observation.labels.first else { continue }
            
            // Only process if YOLO is at least 50% sure
            if topLabel.confidence > 0.50 { 
                
                let confPct = Int(topLabel.confidence * 100)
                network.sendLog("👀 YOLO sees: \(topLabel.identifier) (\(confPct)%)")
                
                let bbox = observation.boundingBox
                let arKitCenter = CGPoint(x: bbox.midX, y: 1.0 - bbox.minY) 
                var finalObjPos: SIMD3<Float>? = nil
                
                // 🚀 STAGE 1: DIRECT LiDAR DEPTH SAMPLING 🚀
                if let lidarDepth = getLiDARDistance(frame: frame, normalizedCenter: arKitCenter) {
                    
                    let intrinsics = frame.camera.intrinsics
                    let fx = intrinsics[0][0]; let fy = intrinsics[1][1]
                    let cx = intrinsics[2][0]; let cy = intrinsics[2][1]
                    
                    let imageRes = frame.camera.imageResolution
                    let ub = Float(arKitCenter.x * imageRes.width)
                    let vb = Float(arKitCenter.y * imageRes.height)
                    
                    let xn = (ub - cx) / fx
                    let yn = (vb - cy) / fy
                    
                    let Xc = xn * lidarDepth
                    let Yc = yn * lidarDepth
                    let Zc = -lidarDepth 
                    
                    let localPoint = simd_float4(Xc, Yc, Zc, 1)
                    let worldPoint = frame.camera.transform * localPoint
                    
                    finalObjPos = SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
                    network.sendLog("🎯 LiDAR mapped \(topLabel.identifier)")
                } 
                // 🧮 STAGE 2: GEOMETRIC MATH FALLBACK (If LiDAR is missing/fails) 🧮
                else {
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
                        network.sendLog("🧮 Math Fallback mapped \(topLabel.identifier)")
                    }
                }
                
                if let objPos = finalObjPos {
                    let distToObject = distance(currentPos, objPos)
                    let formattedDist = String(format: "%.2f", distToObject)
                    
                    self.add3DLabel(text: "\(topLabel.identifier): \(formattedDist)m (\(confPct)%)", position: objPos)
                    
                    let detectionPacket: [String: Any] = [
                        "type": "survey",
                        "timestamp": frame.timestamp,
                        "position": [objPos.x, objPos.y, objPos.z],
                        "label": topLabel.identifier,
                        "note": "Dist: \(formattedDist)m, Conf: \(confPct)%"
                    ]
                    network.sendPose(detectionPacket)
                }
            }
        }
    }

    // --- 🚀 LiDAR SENSOR READING 🚀 ---
    private func getLiDARDistance(frame: ARFrame, normalizedCenter: CGPoint) -> Float? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let depthMap = depthData.depthMap
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        
        let x = Int(normalizedCenter.x * CGFloat(width))
        let y = Int(normalizedCenter.y * CGFloat(height))
        
        let safeX = max(0, min(x, width - 1))
        let safeY = max(0, min(y, height - 1))
        
        if let baseAddress = CVPixelBufferGetBaseAddress(depthMap) {
            let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
            let distance = floatBuffer[safeY * width + safeX]
            if distance > 0 { return distance }
        }
        return nil
    }

    // --- 🚀 OBSTACLE AVOIDANCE 🚀 ---
    private func detectObstacleDistance(frame: ARFrame) -> Float {
        var rawDistance: Float = 10.0 
        let screenCenter = CGPoint(x: 0.5, y: 0.5)
        if let lidarDist = getLiDARDistance(frame: frame, normalizedCenter: screenCenter) {
            rawDistance = lidarDist
        } 
        smoothedObstacleDist = (smoothingAlpha * rawDistance) + ((1.0 - smoothingAlpha) * smoothedObstacleDist)
        return smoothedObstacleDist
    }

    // --- 🌟 3D AR LABELS 🌟 ---
    private func add3DLabel(text: String, position: SIMD3<Float>) {
        DispatchQueue.main.async {
            let textGeometry = SCNText(string: text, extrusionDepth: 0.01)
            textGeometry.font = UIFont.systemFont(ofSize: 1.0)
            textGeometry.firstMaterial?.diffuse.contents = UIColor.green
            let textNode = SCNNode(geometry: textGeometry)
            textNode.scale = SCNVector3(0.05, 0.05, 0.05)
            textNode.position = SCNVector3(position.x, position.y + 0.20, position.z)
            
            let billboardConstraint = SCNBillboardConstraint()
            billboardConstraint.freeAxes = .Y
            textNode.constraints = [billboardConstraint]
            
            self.sceneView.scene.rootNode.addChildNode(textNode)
            
            let fadeOut = SCNAction.fadeOut(duration: 2.0)
            let remove = SCNAction.removeFromParentNode()
            textNode.runAction(SCNAction.sequence([fadeOut, remove]))
        }
    }
}