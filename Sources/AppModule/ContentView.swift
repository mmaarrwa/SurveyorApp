import SwiftUI
import ARKit

struct ContentView: View {
    @StateObject private var arManager = ARManager.shared

    var body: some View {
        ZStack {
            // 1. The Camera View Background
            ARViewContainer(arManager: arManager)
                .edgesIgnoringSafeArea(.all)

            // 2. The UI Overlay
            VStack {
                Spacer()
                
                // Compact Control Panel
                VStack(spacing: 12) {
                    
                    // --- Status Bar ---
                    HStack {
                        Image(systemName: "circle.fill")
                            .foregroundColor(statusColor)
                            .font(.system(size: 10))
                        
                        Text(arManager.statusText)
                            .font(.footnote)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                        
                        Spacer()
                    }
                    .padding(.bottom, 2)
                    
                    // --- ROW 1: Network Connection ---
                    HStack(spacing: 10) {
                        Image(systemName: "wifi")
                            .foregroundColor(.gray)
                            .frame(width: 20)
                        
                        TextField("PC IP", text: $arManager.serverIP)
                            .textFieldStyle(.plain)
                            .font(.subheadline)
                            .keyboardType(.decimalPad) // Perfect for IPs (Numbers + Dot)
                            .padding(8)
                            .background(Color.white.opacity(0.2))
                            .cornerRadius(6)
                            .foregroundColor(.white)
                        
                        Button(action: {
                            hideKeyboard()
                            arManager.connectToNetwork()
                        }) {
                            Text("Connect")
                                .font(.caption)
                                .bold()
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(6)
                        }
                    }
                    
                    // --- ROW 2: Height & Start/Stop ---
                    HStack(spacing: 10) {
                        Image(systemName: "ruler")
                            .foregroundColor(.gray)
                            .frame(width: 20)
                        
                        TextField("0.20", text: $arManager.cameraHeightInput)
                            .textFieldStyle(.plain)
                            .font(.subheadline)
                            .keyboardType(.decimalPad)
                            .padding(8)
                            .background(Color.white.opacity(0.2))
                            .cornerRadius(6)
                            .foregroundColor(.white)
                            .frame(width: 60)
                        
                        Text("m")
                            .font(.caption)
                            .foregroundColor(.gray)
                            
                        Spacer()
                            
                        Button(action: {
                            hideKeyboard()
                            arManager.toggleStreaming()
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: arManager.isStreaming ? "stop.fill" : "play.fill")
                                Text(arManager.isStreaming ? "STOP" : "START")
                            }
                            .font(.caption)
                            .bold()
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(arManager.isStreaming ? Color.red : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(6)
                        }
                    }
                }
                .padding(16)
                .background(Color.black.opacity(0.85)) // Darker, sleeker background
                .cornerRadius(16)
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
                .shadow(radius: 10)
            }
        }
        .onAppear {
            // Wakes up the camera so you aren't staring at a black screen
            arManager.startSessionIfNeeded()
        }
    }
    
    // Helper to dynamically change the little status dot color
    private var statusColor: Color {
        if arManager.isStreaming { return .green }
        if arManager.statusText.contains("Connected") { return .blue }
        return .orange
    }
    
    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

struct ARViewContainer: UIViewRepresentable {
    var arManager: ARManager
    func makeUIView(context: Context) -> ARSCNView { return arManager.sceneView }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}