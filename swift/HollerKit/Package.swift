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
        .package(path: "../../../mlx-audio-swift"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMajor(from: "3.31.3")),
    ],
    targets: [
        .target(
            name: "HollerKit",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/HollerKit"
        ),
        .executableTarget(
            name: "HollerCLI",
            dependencies: ["HollerKit"],
            path: "Sources/HollerCLI"
        ),
        .testTarget(
            name: "HollerKitTests",
            dependencies: ["HollerKit"],
            path: "Tests/HollerKitTests"
        ),
    ]
)
