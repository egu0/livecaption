// CaptionState.swift — @Observable state owned by the main actor.
// Drives the SwiftUI ContentView; mutated exclusively from MainActor callbacks.
import Foundation
import Observation

@MainActor
@Observable
final class CaptionState {
    var statusLines: [StatusLine] = []
    var finalLines: [FinalLine] = []
    var partialText: String? = nil

    func applyStatus(_ message: String) {
        let lower = message.lowercased()
        // Suppress model-loading chatter and the persistent "Listening" indicator;
        // errors and lifecycle transitions (Stopping…) still appear in the transcript.
        if lower.contains("loading") || lower.hasPrefix("asr:") || lower.contains("listening") {
            return
        }
        let isError = lower.hasPrefix("error") || lower.hasPrefix("fatal")
        statusLines.append(StatusLine(
            text: message, isError: isError, isActive: false
        ))
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

struct StatusLine: Identifiable {
    let id = UUID()
    let text: String
    let isError: Bool
    let isActive: Bool
}
