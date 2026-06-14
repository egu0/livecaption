// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "livecaption-window",
    platforms: [.macOS("14.2")],
    products: [
        .executable(name: "livecaption-window", targets: ["CaptionWindow"])
    ],
    targets: [
        .executableTarget(
            name: "CaptionWindow",
            path: "Sources/CaptionWindow"
        )
    ]
)
