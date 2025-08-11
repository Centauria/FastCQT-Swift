// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FastCQT",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "FastCQT",
            targets: ["FastCQT"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", .upToNextMajor(from: "1.0.3")),
        .package(url: "https://github.com/dclelland/Plinth.git", from: "2.0.0"),
        .package(url: "https://github.com/jkl1337/SwiftPFFFT.git", .upToNextMajor(from: "0.1.1")),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .binaryTarget(
            name: "CSoxr",
            path: "Sources/soxr.xcframework"
        ),
        .target(
            name: "Soxr",
            dependencies: ["CSoxr"],
            sources: ["dummy.c"]
        ),
        .target(
            name: "STFT",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Plinth", package: "Plinth"),
                .product(name: "PFFFT", package: "SwiftPFFFT"),
            ]
        ),
        .target(
            name: "FastCQT",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Plinth", package: "Plinth"),
                "STFT",
                "Soxr",
            ]
        ),
        .testTarget(
            name: "FastCQTTests",
            dependencies: ["FastCQT"]),
    ]
)
