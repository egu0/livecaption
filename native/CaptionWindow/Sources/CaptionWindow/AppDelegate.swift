// AppDelegate.swift — NSApplicationDelegate that creates the floating caption window.
import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let state: CaptionState
    private var window: NSWindow?
    private var eventMonitor: Any?
    private var escapeCloseGate = EscapeCloseGate(confirmInterval: 1.0)

    init(state: CaptionState) {
        self.state = state
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let size = NSSize(width: 800, height: 100)

        // Build a borderless floating window — no title bar, full content
        // area. Draggable by background, resizable, with standard shadow.
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless, .resizable],
            backing: .buffered,
            defer: false
        )
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.delegate = self
        window.isMovableByWindowBackground = true
        window.hasShadow = true
        window.isOpaque = false
        window.backgroundColor = .clear
        window.minSize = NSSize(width: 200, height: 40)
        window.contentMinSize = size
        // .floating windows don't get an app dock tile; make one so Cmd-Tab works
        NSApp.setActivationPolicy(.regular)

        // Rounded-corner container — a plain NSView clips cleanly without
        // fighting the window server's vibrancy compositing.
        let container = NSView(frame: NSRect(origin: .zero, size: size))
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
        hostingView.setContentHuggingPriority(.defaultLow, for: .vertical)
        hostingView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        hostingView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        hostingView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
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
        // Anchor the hosting view to a minimum width so the empty SwiftUI
        // transcript doesn't collapse the borderless window on first display.
        hostingView.widthAnchor.constraint(greaterThanOrEqualToConstant: size.width).isActive = true
        // Re-assert size, then center. Order matters: setFrame* after
        // contentView swap prevents borderless windows from sizing to
        // content fittingSize; center after setFrame so the origin sticks.
        window.setContentSize(size)
        window.center()
        self.window = window

        // ESC twice within the confirmation interval → terminate
        eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == 53 {  // ESC
                if self?.escapeCloseGate.shouldClose(at: event.timestamp) == true {
                    self?.window?.close()
                }
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
