import SwiftUI
import ARKit

struct ContentView: View {
    @StateObject private var arManager = ARManager.shared

    var body: some View {
        ZStack {
            ARViewContainer(arManager: arManager)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                
                VStack(spacing: 12) {
                    Text(arManager.statusText)
                        .font(.headline)
                        .padding(8)
                        .frame(maxWidth: .infinity)
                        .background(Color.black.opacity(0.5))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    
                    // ROW 1: Network Connection
                    HStack {
                        Text("IP:")
                            .foregroundColor(.white)
                            .bold()
                        TextField("PC IP", text: $arManager.serverIP)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.numbersAndPunctuation)
                        
                        Button(action: {
                            hideKeyboard()
                            arManager.connectToNetwork()
                        }) {
                            Text("Connect")
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                    
                    // ROW 2: Height and Start/Stop
                    HStack {
                        Text("Cam Height (m):")
                            .foregroundColor(.white)
                            .bold()
                        
                        TextField("0.20", text: $arManager.cameraHeightInput)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.decimalPad)
                            .frame(width: 70)
                            
                        Spacer()
                            
                        Button(action: {
                            hideKeyboard()
                            arManager.toggleStreaming()
                        }) {
                            Text(arManager.isStreaming ? "STOP" : "START")
                                .font(.headline)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 8)
                                .background(arManager.isStreaming ? Color.red : Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.75))
                .cornerRadius(15)
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
            }
        }
        // Removed the .onAppear autostart so the camera stays off until you are ready!
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