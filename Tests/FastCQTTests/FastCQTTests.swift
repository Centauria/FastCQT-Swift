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
        let m = Int.random(in: 2...1000)
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

    func testParabolicInterpolationArray() throws {
        let x: [Float] = [0, 1, 2, 0, 1, 2, -1, 0, 2, 1]
        let y = parabolicInterpolation(x)
        let result: [Float] = [0, 0, -0.16666667, 0.16666667, 0, -0.25, 0.25, 0, 0.16666667, 0]
        assert(y == result)
    }

    func testParabolicInterpolationMatrix() throws {
        let x: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                0.84539491, 0.31397233, 0.99722973, 0.45916594, 0.53021606,
                0.43493118, 0.85664736, 0.01921022, 0.75724459, 0.15218493,
                0.54920612, 0.98290162, 0.91595992, 0.17742043, 0.45801586,
                0.43670925, 0.22165366, 0.07434987, 0.10104737, 0.54333287,
            ])
        let y0 = parabolicInterpolation(x)
        let result0: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                0, 0, 0, 0, 0,
                0.28222505, 0.80318915, 0.02167462, -0.1604651, 0.05278857,
                0.00392041, -0.35774203, 0.01585968, 0.65169907, 0.88690079,
                0, 0, 0, 0, 0,
            ]
        )
        assert((y0 - result0).absolute().maximum() < 1e-7)
        let y1 = parabolicInterpolation(x, axis: 1)
        let result1: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                0, -0.06249992, 0.05944121, 0.38335496, 0,
                0, -0.16507956, 0.03154699, 0.04950313, 0,
                0, 0.366287, -0.59967529, 0.22467293, 0,
                0, 0, 0.34656726, -0.56424029, 0,
            ]
        )
        assert((y1 - result1).absolute().maximum() < 1e-7)
    }

    func testGradient() throws {
        let x: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                0.84539491, 0.31397233, 0.99722973, 0.45916594, 0.53021606,
                0.43493118, 0.85664736, 0.01921022, 0.75724459, 0.15218493,
                0.54920612, 0.98290162, 0.91595992, 0.17742043, 0.45801586,
                0.43670925, 0.22165366, 0.07434987, 0.10104737, 0.54333287,
            ])
        let y0 = gradient(y: x)
        let result0: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                -4.10463740e-01, 5.42675033e-01, -9.78019511e-01, 2.98078644e-01, -3.78031130e-01,
                -1.48094398e-01, 3.34464647e-01, -4.06349053e-02, -1.40872757e-01, -3.61000988e-02,
                8.89039246e-04, -3.17496847e-01, 2.75698290e-02, -3.28098609e-01, 1.95573972e-01,
                -1.12496864e-01, -7.61247956e-01, -8.41610042e-01, -7.63730608e-02, 8.53170104e-02,
            ]
        )
        assert((y0 - result0).absolute().maximum() < 1e-7)
        let y1 = gradient(y: x, axis: 1)
        let result1: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                -0.53142259, 0.07591741, 0.07259681, -0.23350684, 0.07105011,
                0.42171618, -0.20786048, -0.04970139, 0.06648735, -0.60505966,
                0.4336955, 0.1833769, -0.4027406, -0.22897203, 0.28059543,
                -0.21505559, -0.18117969, -0.06030315, 0.2344915, 0.4422855,
            ]
        )
        assert((y1 - result1).absolute().maximum() < 1e-7)
    }

    func testMaximumByAxis() throws {
        let x: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 5),
            elements: [
                0.84539491, 0.31397233, 0.99722973, 0.45916594, 0.53021606,
                0.43493118, 0.85664736, 0.01921022, 0.75724459, 0.15218493,
                0.54920612, 0.98290162, 0.91595992, 0.17742043, 0.45801586,
                0.43670925, 0.22165366, 0.07434987, 0.10104737, 0.54333287,
            ])
        let y0 = x.maximum(axis: 0)
        let result0: Matrix<Float> = .init(
            shape: .init(rows: 1, columns: 5),
            elements: [
                0.84539491, 0.98290162, 0.99722973, 0.75724459, 0.54333287,
            ]
        )
        assert(y0 == result0)
        let y1 = x.maximum(axis: 1)
        let result1: Matrix<Float> = .init(
            shape: .init(rows: 4, columns: 1),
            elements: [
                0.99722973, 0.85664736, 0.98290162, 0.54333287,
            ]
        )
        assert(y1 == result1)
    }
}
