// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "HollerTest",
    platforms: [.macOS("15.0")],
    dependencies: [
        .package(path: "../speech-swift")
    ],
    targets: [
        .executableTarget(
            name: "HollerTest",
            dependencies: [
                .product(name: "Qwen3TTS", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ],
            path: "Sources"
        )
    ]
)
