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
