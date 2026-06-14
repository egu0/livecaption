// ContentView.swift — SwiftUI view hierarchy: status bar + scrollable transcript.
import SwiftUI

struct ContentView: View {
    @State private var state: CaptionState

    init(state: CaptionState) {
        self._state = State(initialValue: state)
    }

    var body: some View {
        VStack(spacing: 0) {
            // ---- Status bar ----
            HStack {
                Text(state.statusMessage)
                    .font(.system(size: 13))
                    .foregroundColor(colorForStatus)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // ---- Transcript area (top-to-bottom, text selectable) ----
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(state.finalLines) { line in
                            Text(line.text)
                                .font(.system(size: 16))
                                .foregroundColor(.primary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if let partial = state.partialText {
                            Text(partial)
                                .font(.system(size: 16))
                                .foregroundColor(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id("partial-anchor")
                        }
                    }
                    .padding(12)
                }
                .onChange(of: state.finalLines.count) {
                    proxy.scrollTo("partial-anchor", anchor: .bottom)
                }
                .onChange(of: state.partialText) {
                    proxy.scrollTo("partial-anchor", anchor: .bottom)
                }
            }
        }
    }

    private var colorForStatus: Color {
        let msg = state.statusMessage.lowercased()
        if msg.hasPrefix("error") || msg.hasPrefix("fatal") {
            return .red
        }
        if msg.contains("listening") {
            return .green
        }
        return .secondary
    }
}
