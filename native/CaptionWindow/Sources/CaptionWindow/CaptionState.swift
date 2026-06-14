// CaptionState.swift — @Observable state owned by the main actor.
// Drives the SwiftUI ContentView; mutated exclusively from MainActor callbacks.
import Foundation
import Observation

@MainActor
@Observable
final class CaptionState {
    var statusMessage: String = "Starting…"
    var finalLines: [FinalLine] = []
    var partialText: String? = nil

    func applyStatus(_ message: String) {
        statusMessage = message
    }

    func applyPartial(text: String) {
        partialText = text
    }

    func applyFinal(text: String) {
        finalLines.append(FinalLine(text: text))
        partialText = nil
    }
}

struct FinalLine: Identifiable {
    let id = UUID()
    let text: String
}
