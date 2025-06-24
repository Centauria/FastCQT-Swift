import Accelerate
import Numerics
import Plinth
import XCTest

@testable import FastCQT

final class FastCQTTests: XCTestCase {
    func testExample() throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
    }

    func testDivCeil() throws {
        assert(divceil(10, 5) == 2)
        assert(divceil(11, 5) == 3)
    }

    func testSparseMultiply() throws {
        let rowCount = Int32(4)
        let columnCount = Int32(3)
        let blockCount = 5
        let blockSize = UInt8(1)
        let rowIndices: [Int32] = [0, 3, 1, 0, 3]
        let columnIndices: [Int32] = [0, 0, 1, 2, 2]
        let dataReal: [Float] = [7, 3, 1, 2, -1]

        let A = SparseConvertFromCoordinate(
            rowCount, columnCount,
            blockCount, blockSize,
            .init(),
            rowIndices, columnIndices,
            dataReal)
        defer {
            SparseCleanup(A)
        }

        let X: Matrix<Float> = .init(shape: .init(rows: 5, columns: 4)) { i, j in
            .init(i * 4 + j + 1)
        }

        let Y = SparseMultiply(X, A)

        let result: Matrix<Float> = .init(
            shape: .init(rows: 5, columns: 3),
            elements: [
                19, 2, -2,
                59, 6, 2,
                99, 10, 6,
                139, 14, 10,
                179, 18, 14,
            ]
        )
        assert(Y == result)
    }

    func testSparseMultiply2() throws {
        let rowCount = Int32(400)
        let columnCount = Int32(3)
        let blockCount = 9
        let blockSize = UInt8(1)
        let rowIndices: [Int32] = [2, 58, 303, 58, 70, 225, 10, 59, 111]
        let columnIndices: [Int32] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        let dataReal: [Float] = [-4, 1, 2, -1, -1, 1, 0.5, 0.5, -1]

        let A = SparseConvertFromCoordinate(
            rowCount, columnCount,
            blockCount, blockSize,
            .init(),
            rowIndices, columnIndices,
            dataReal)
        defer {
            SparseCleanup(A)
        }

        let X: Matrix<Float> = .init(shape: .init(rows: 24, columns: 400)) { i, j in
            Float(i) / 24 * (Float(j) / 400 - 0.5) - Float(j) * 0.01
        }

        let Y = SparseMultiply(X, A)

        let result: Matrix<Float> = .init(
            shape: .init(rows: 24, columns: 3),
            elements: [
                -6.56, -0.97, 0.76500005,
                -6.4708333, -0.9390625, 0.75703126,
                -6.3816667, -0.90812516, 0.74906254,
                -6.2925, -0.8771875, 0.74109375,
                -6.203333, -0.84624994, 0.7331251,
                -6.1141667, -0.8153126, 0.7251563,
                -6.025, -0.78437495, 0.7171875,
                -5.935833, -0.7534375, 0.70921886,
                -5.846667, -0.7225001, 0.70125,
                -5.7575, -0.69156253, 0.6932813,
                -5.668333, -0.660625, 0.6853125,
                -5.579167, -0.62968755, 0.6773437,
                -5.49, -0.5987501, 0.66937506,
                -5.400833, -0.56781244, 0.6614063,
                -5.311667, -0.5368751, 0.6534375,
                -5.2225, -0.5059376, 0.6454688,
                -5.133333, -0.4749999, 0.6375,
                -5.0441666, -0.4440626, 0.6295312,
                -4.955, -0.41312504, 0.62156254,
                -4.8658333, -0.38218737, 0.61359376,
                -4.7766666, -0.35125017, 0.6056251,
                -4.6875, -0.3203125, 0.59765625,
                -4.598333, -0.28937495, 0.5896876,
                -4.5091667, -0.25843763, 0.5817188,
            ]
        )
        assert((Y - result).absolute().maximum() < 1e-7)
    }

    func testRandomSparseMultiply() throws {
        for _ in 1...10 {
            let m = Int.random(in: 2...30000)
            let n = Int.random(in: 2...1000)
            let d = Int.random(in: 2...100)
            let blockCount = Int.random(in: 3...d * n / 10)
            let blockSize = UInt8(1)
            var index = Array(0..<d * n)
            index.shuffle()
            let rowIndices: [Int32] = index.prefix(blockCount).map { Int32($0 % d) }
            let columnIndices: [Int32] = index.prefix(blockCount).map { Int32($0 / d) }
            let dataReal: [Float] = rowIndices.map { _ in Float.random(in: -2...2) }

            let A = SparseConvertFromCoordinate(
                Int32(d), Int32(n),
                blockCount, blockSize,
                .init(),
                rowIndices, columnIndices,
                dataReal)
            defer {
                SparseCleanup(A)
            }

            let X: Matrix<Float> = .random(shape: .init(rows: m, columns: d), in: -1...1)

            let Y = SparseMultiply(X, A)
            let result = SparseMultiplyNaïve(X, A)

            assert((Y - result).absolute().maximum() < 1e-5)
        }
    }

    func testComplexSparseMultiply() throws {
        let rowCount = Int32(4)
        let columnCount = Int32(3)
        let blockCount = 5
        let blockSize = UInt8(1)
        let rowIndices: [Int32] = [0, 3, 1, 2, 3]
        let columnIndices: [Int32] = [0, 0, 1, 2, 2]
        let dataReal: [Float] = [1.0, 3.0, 0.0, 3, 0]
        let dataImag: [Float] = [2, 4, 1, 0, -1]

        /// The _A_ in _Y=AX_.
        let real = SparseConvertFromCoordinate(
            rowCount, columnCount,
            blockCount, blockSize,
            .init(),
            rowIndices, columnIndices,
            dataReal)
        defer {
            SparseCleanup(real)
        }
        let A: SparseMatrix_ComplexFloat = .init(_real: real, _imag: dataImag)

        /// The values for _X_ in _Y=AX_.
        let X: ComplexMatrix<Float> = .init(shape: .init(rows: 5, columns: 4)) { i, j in
            .init(Float(j + 1), Float(i + 1))
        }

        /// The values for _Y_ in _Y=AX_.
        let Y = SparseMultiply(X, A)

        let result: ComplexMatrix<Float> = .init(
            real: .init(
                shape: .init(rows: 5, columns: 3),
                elements: [
                    7, -1, 10,
                    1, -2, 11,
                    -5, -3, 12,
                    -11, -4, 13,
                    -17, -5, 14,
                ]),
            imaginary: .init(
                shape: .init(rows: 5, columns: 3),
                elements: [
                    22, 2, -1,
                    26, 2, 2,
                    30, 2, 5,
                    34, 2, 8,
                    38, 2, 11,
                ]
            ))
        assert(Y == result)
    }
}
