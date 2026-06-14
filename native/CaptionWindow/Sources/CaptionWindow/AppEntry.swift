// main.swift — @main entry point: creates AppDelegate, starts stdin reader, enters run loop.
import AppKit

@main
struct CaptionWindowApp {
    @MainActor
    static func main() {
        let state = CaptionState()
        let delegate = AppDelegate(state: state)
        NSApplication.shared.delegate = delegate

        // Start reading stdin events on a background queue
        startStdinReader(state: state)

        // Enter the run loop; exits when terminate() is called
        NSApplication.shared.run()
    }
}
