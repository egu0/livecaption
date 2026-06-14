// AppDelegate.swift — NSApplicationDelegate that creates the floating caption window.
import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let state: CaptionState
    private var window: NSWindow?
    private var eventMonitor: Any?

    init(state: CaptionState) {
        self.state = state
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Build a borderless floating window — no title bar, full content
        // area. Draggable by background, resizable, with standard shadow.
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 120),
            styleMask: [.borderless, .resizable],
            backing: .buffered,
            defer: false
        )
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.delegate = self
        window.center()
        window.isMovableByWindowBackground = true
        window.hasShadow = true
        window.isOpaque = false
        window.backgroundColor = .clear
        // Minimum size: wide enough to be legible, tall enough to show the status bar
        window.minSize = NSSize(width: 200, height: 40)
        // .floating windows don't get an app dock tile; make one so Cmd-Tab works
        NSApp.setActivationPolicy(.regular)

        // Vibrancy backing
        let visualEffect = NSVisualEffectView()
        visualEffect.blendingMode = .behindWindow
        visualEffect.state = .active
        visualEffect.material = .hudWindow

        // Host SwiftUI content
        let hostingView = NSHostingView(rootView: ContentView(state: state))
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        // Let the window frame dictate size, not the SwiftUI intrinsic size.
        // Both hugging (prefers smaller) and compression-resistance (refuses
        // to shrink below intrinsic) must be lowered so a short contentRect
        // height actually takes effect.
        hostingView.setContentHuggingPriority(.defaultLow, for: .vertical)
        hostingView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        hostingView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        visualEffect.addSubview(hostingView)
        NSLayoutConstraint.activate([
            hostingView.topAnchor.constraint(equalTo: visualEffect.topAnchor),
            hostingView.leadingAnchor.constraint(equalTo: visualEffect.leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: visualEffect.trailingAnchor),
            hostingView.bottomAnchor.constraint(equalTo: visualEffect.bottomAnchor),
        ])

        window.contentView = visualEffect
        // Clip the entire window (including the vibrancy blur pass) to
        // system-consistent rounded corners.  Applying cornerRadius to the
        // visualEffect view alone only clips subviews — the .behindWindow
        // blur is composited by the window server and ignores view-layer
        // masking, which left a white rectangular halo behind the curves.
        window.contentView?.wantsLayer = true
        window.contentView?.layer?.cornerRadius = 12
        window.contentView?.layer?.masksToBounds = true
        self.window = window

        // ESC key → terminate
        eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == 53 {  // ESC
                self?.window?.close()
                return nil
            }
            return event
        }

        window.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func windowWillClose(_ notification: Notification) {
        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
            eventMonitor = nil
        }
        NSApplication.shared.terminate(nil)
    }
}
