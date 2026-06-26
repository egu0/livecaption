import Foundation

struct EscapeCloseGate {
    let confirmInterval: TimeInterval
    private var lastEscapeAt: TimeInterval?

    init(confirmInterval: TimeInterval) {
        self.confirmInterval = confirmInterval
    }

    mutating func shouldClose(at timestamp: TimeInterval) -> Bool {
        defer { lastEscapeAt = timestamp }

        guard let lastEscapeAt else {
            return false
        }
        return timestamp - lastEscapeAt <= confirmInterval
    }
}
