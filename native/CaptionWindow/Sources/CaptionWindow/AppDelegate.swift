// AppDelegate.swift — NSApplicationDelegate that creates the floating caption window.
import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let state: CaptionState
    private var window: NSWindow?
    private var eventMonitor: Any?

    private static let savedFrameKey = "captionWindowFrame"

    init(state: CaptionState) {
        self.state = state
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Restore saved frame, or use a sensible default
        let defaultFrame = NSRect(x: 0, y: 0, width: 800, height: 100)
        let frame: NSRect = {
            if let saved = UserDefaults.standard.string(forKey: Self.savedFrameKey) {
                let rect = NSRectFromString(saved)
                // Guard against zero / negative dimensions (corrupted save)
                if rect.size.width >= 200 && rect.size.height >= 40 {
                    return rect
                }
            }
            return defaultFrame
        }()

        // Build a borderless floating window — no title bar, full content
        // area. Draggable by background, resizable, with standard shadow.
        let window = NSWindow(
            contentRect: frame,
            styleMask: [.borderless, .resizable],
            backing: .buffered,
            defer: false
        )
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.delegate = self
        if UserDefaults.standard.string(forKey: Self.savedFrameKey) == nil {
            window.center()  // first launch only — center on screen
        }
        window.isMovableByWindowBackground = true
        window.hasShadow = true
        window.isOpaque = false
        window.backgroundColor = .clear
        // Minimum size: wide enough to be legible, tall enough to show the status bar
        window.minSize = NSSize(width: 200, height: 40)
        // .floating windows don't get an app dock tile; make one so Cmd-Tab works
        NSApp.setActivationPolicy(.regular)

        // Rounded-corner container — a plain NSView clips cleanly without
        // fighting the window server's vibrancy compositing
        let container = NSView()
        container.wantsLayer = true
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true

        // Vibrancy backing — fills the container to get rounded corners
        let visualEffect = NSVisualEffectView()
        visualEffect.blendingMode = .behindWindow
        visualEffect.state = .active
        visualEffect.material = .hudWindow
        visualEffect.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(visualEffect)

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
            visualEffect.topAnchor.constraint(equalTo: container.topAnchor),
            visualEffect.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            visualEffect.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            visualEffect.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            hostingView.topAnchor.constraint(equalTo: visualEffect.topAnchor),
            hostingView.leadingAnchor.constraint(equalTo: visualEffect.leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: visualEffect.trailingAnchor),
            hostingView.bottomAnchor.constraint(equalTo: visualEffect.bottomAnchor),
        ])

        window.contentView = container
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
        // Persist the current frame so the next launch restores it
        if let w = window {
            UserDefaults.standard.set(NSStringFromRect(w.frame), forKey: Self.savedFrameKey)
        }
        NSApplication.shared.terminate(nil)
    }
}
