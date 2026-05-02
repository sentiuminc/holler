// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "HollerKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "HollerKit", targets: ["HollerKit"]),
        .executable(name: "holler", targets: ["HollerCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/sentiuminc/mlx-audio-swift.git", from: "0.31.3-holler.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
    ],
    targets: [
        .target(
            name: "HollerKit",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "HollerCLI",
            dependencies: ["HollerKit"]
        ),
        .testTarget(
            name: "HollerKitTests",
            dependencies: ["HollerKit"]
        ),
    ]
)
