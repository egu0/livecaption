// StdinReader.swift — reads JSON Lines from stdin via readabilityHandler,
// decodes StdinEvent values, and dispatches them to CaptionState on the main actor.
import AppKit
import Foundation

enum StdinEvent: Decodable {
    case status(message: String)
    case partial(text: String)
    case final(text: String)

    enum CodingKeys: String, CodingKey {
        case type, message, text
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "status":
            self = .status(message: try container.decode(String.self, forKey: .message))
        case "partial":
            self = .partial(text: try container.decode(String.self, forKey: .text))
        case "final":
            self = .final(text: try container.decode(String.self, forKey: .text))
        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.type],
                    debugDescription: "Unknown event type: \(type)"
                )
            )
        }
    }
}

func startStdinReader(state: CaptionState) {
    let handle = FileHandle.standardInput
    let decoder = JSONDecoder()
    var buffer = Data()

    handle.readabilityHandler = { fh in
        let chunk = fh.availableData
        guard !chunk.isEmpty else {
            // EOF: Python parent stopped writing → exit
            DispatchQueue.main.async {
                NSApplication.shared.terminate(nil)
            }
            handle.readabilityHandler = nil
            return
        }
        buffer.append(chunk)

        // Split buffer into complete lines (LF-delimited)
        let newline = Data([0x0A])
        while let range = buffer.range(of: newline) {
            let lineData = buffer.subdata(in: 0..<range.lowerBound)
            buffer.removeSubrange(0...range.lowerBound)

            guard
                let line = String(data: lineData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !line.isEmpty,
                let data = line.data(using: .utf8)
            else { continue }

            do {
                let event = try decoder.decode(StdinEvent.self, from: data)
                DispatchQueue.main.async {
                    switch event {
                    case .status(let message):
                        state.applyStatus(message)
                    case .partial(let text):
                        state.applyPartial(text: text)
                    case .final(let text):
                        state.applyFinal(text: text)
                    }
                }
            } catch {
                // Skip malformed lines — may be partial writes during shutdown
            }
        }
    }
}
