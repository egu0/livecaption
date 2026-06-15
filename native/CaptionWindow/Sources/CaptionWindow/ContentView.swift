// ContentView.swift — SwiftUI view hierarchy: scrollable transcript with inline status lines.
import SwiftUI

struct ContentView: View {
    @State private var state: CaptionState

    init(state: CaptionState) {
        self._state = State(initialValue: state)
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(state.statusLines) { line in
                        Text(line.text)
                            .font(.system(size: 12, weight: line.isError ? .medium : .regular))
                            .foregroundColor(colorForStatus(line))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
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
            .onChange(of: state.statusLines.count) {
                proxy.scrollTo("partial-anchor", anchor: .bottom)
            }
        }
    }

    private func colorForStatus(_ line: StatusLine) -> Color {
        if line.isError {
            return .red
        }
        if line.isActive {
            return .green
        }
        return .secondary
    }
}
