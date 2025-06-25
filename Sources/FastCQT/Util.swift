import Accelerate
import Foundation
import Numerics
import Plinth

public func divceil(_ a: Int, _ b: Int) -> Int {
    let d = div(Int32(a), Int32(b))
    return Int(d.quot + (d.rem > 0 ? 1 : 0))
}

@inline(__always)
public func numTwoFactors(_ x: Int) -> Int {
    guard x > 0 else { return 0 }
    return x.trailingZeroBitCount
}

public func phasor(angles: [Float]) -> ComplexMatrix<Float> {
    let n = angles.count
    var u = [Float](repeating: 0, count: n)
    var v = [Float](repeating: 0, count: n)
    vForce.sincos(angles, sinResult: &v, cosResult: &u)
    let z: ComplexMatrix<Float> = .init(
        real: .init(shape: .row(length: n), elements: u),
        imaginary: .init(shape: .row(length: n), elements: v))
    return z
}

public func phasor2(angles: [Float32]) -> [Complex<Float32>] {
    return angles.map { a in
        Complex.exp(.init(0, a))
    }
}

public func normalize(S: ComplexMatrix<Float>, norm: Float = 1) -> ComplexMatrix<Float> {
    let nrows = S.shape.rows
    let ncols = S.shape.columns
    var Snorm: ComplexMatrix<Float>

    let mag = S.absolute() ** norm
    if nrows == 1 || ncols == 1 {
        let length = Float.pow(mag.sum(), 1 / norm)
        Snorm = S / length
    } else {
        let magnSum: Matrix<Float> = .init(
            shape: .row(length: ncols),
            { (_: Int, i: Int) -> Float in
                mag[0..<nrows, i].sum()
            })
        let length = magnSum ** (1 / norm)
        let lr = length.repeated(rows: nrows)
        Snorm = S / lr
    }
    return Snorm
}

public func sparsify_rows(
    x: ComplexMatrix<Float>, quantile: Float = 0.01
) -> SparseMatrix_ComplexFloat {
    let nrows = x.shape.rows
    let ncols = x.shape.columns

    let xi: ComplexMatrix<Float>

    if ncols == 1 {
        xi = x.transposed()
    } else {
        xi = x
    }

    let mags = xi.absolute()
    var magSort: Matrix<Float> = .init(shape: mags.shape, elements: mags.elements)
    var magNorm: Matrix<Float> = .zeros(shape: mags.shape)

    (0..<nrows).forEach { i in
        let mag = magSort[i, 0..<ncols]
        let norm = mag.sum()
        magSort[i, 0..<ncols] = mag.sort(.ascending)
        magNorm[i, 0..<ncols] = magSort[i, 0..<ncols] / norm
    }
    // var accumulated: Matrix<Float> = .zeros(shape: .column(length: nrows))
    // (0..<ncols).forEach { i in
    //     accumulated += mags[0..<nrows, i]
    //     mags[0..<nrows, i] = accumulated
    // }
    var reals: [Float] = []
    var imags: [Float] = []
    var rowIndices: [Int32] = []
    var columnIndices: [Int32] = []
    for i in 0..<nrows {
        var acc: Float = 0
        var mark: Float = 0
        for j in 0..<ncols {
            acc += magNorm[i, j]
            if acc >= quantile {
                mark = magSort[i, j]
                break
            }
        }
        for k in 0..<ncols {
            if mags[i, k] >= mark {
                let z = xi[i, k]
                reals.append(z.real)
                imags.append(z.imaginary)
                rowIndices.append(Int32(k))
                columnIndices.append(Int32(i))
            }
        }
    }
    return .init(
        _real: SparseConvertFromCoordinate(
            Int32(ncols), Int32(nrows),
            reals.count, UInt8(1), .init(),
            &rowIndices, &columnIndices, &reals),
        _imag: imags
    )
}

public func gradient(y: Matrix<Float>, axis: Int = 0) -> Matrix<Float> {
    precondition((0...1).contains(axis), "axis must be 0(row) or 1(column)")
    let row = y.shape.rows
    let col = y.shape.columns
    let g: Matrix<Float>
    if axis == 0 {
        g = .init(
            shape: y.shape,
            elements: [Float](unsafeUninitializedCapacity: row * col) {
                ptr, count in
                let output = ptr.baseAddress!
                y.elements.withUnsafeBufferPointer {
                    let input = $0.baseAddress!
                    vDSP_vsub(
                        input, 1,
                        input.advanced(by: col), 1,
                        output, 1,
                        vDSP_Length(col))
                    for i in 1..<row - 1 {
                        vDSP_vsub(
                            input.advanced(by: (i - 1) * col), 1,
                            input.advanced(by: (i + 1) * col), 1,
                            output.advanced(by: i * col), 1,
                            vDSP_Length(col))
                    }
                    vDSP_vsub(
                        input.advanced(by: (row - 2) * col), 1,
                        input.advanced(by: (row - 1) * col), 1,
                        output.advanced(by: (row - 1) * col), 1,
                        vDSP_Length(col))
                    var b: Float = 2
                    vDSP_vsdiv(
                        output.advanced(by: col), 1,
                        &b,
                        output.advanced(by: col), 1,
                        vDSP_Length((row - 2) * col))
                }
                count = row * col
            })
    } else {
        g = .init(
            shape: y.shape,
            elements: [Float](unsafeUninitializedCapacity: row * col) {
                ptr, count in
                let output = ptr.baseAddress!
                y.elements.withUnsafeBufferPointer {
                    let input = $0.baseAddress!
                    var b: Float = 2
                    vDSP_vsub(
                        input, col,
                        input.advanced(by: 1), col,
                        output, col,
                        vDSP_Length(row))
                    for i in 1..<col - 1 {
                        vDSP_vsub(
                            input.advanced(by: i - 1), col,
                            input.advanced(by: i + 1), col,
                            output.advanced(by: i), col,
                            vDSP_Length(row))
                        vDSP_vsdiv(
                            output.advanced(by: i), col,
                            &b,
                            output.advanced(by: i), col,
                            vDSP_Length(row))
                    }
                    vDSP_vsub(
                        input.advanced(by: col - 2), col,
                        input.advanced(by: col - 1), col,
                        output.advanced(by: col - 1), col,
                        vDSP_Length(row))
                }
                count = row * col
            })
    }
    return g
}

extension Matrix {
    @inlinable public mutating func fmapInplace(_ function: (inout [Scalar]) -> Void) {
        function(&elements)
    }
}

extension Matrix where Scalar == Float {
    public func maximum(axis: Int) -> Matrix {
        precondition((0...1).contains(axis))
        let m = shape.rows
        let n = shape.columns
        return if axis == 0 {
            .init(
                shape: .row(length: n),
                elements: [Scalar](unsafeUninitializedCapacity: n) {
                    yptr, count in
                    let y = yptr.baseAddress!
                    elements.withUnsafeBufferPointer {
                        let x = $0.baseAddress!
                        for i in 0..<n {
                            vDSP_maxv(x.advanced(by: i), n, y.advanced(by: i), vDSP_Length(m))
                        }
                    }
                    count = n
                }
            )
        } else {
            .init(
                shape: .column(length: m),
                elements: [Scalar](unsafeUninitializedCapacity: m) {
                    yptr, count in
                    let y = yptr.baseAddress!
                    elements.withUnsafeBufferPointer {
                        let x = $0.baseAddress!
                        for i in 0..<m {
                            vDSP_maxv(x.advanced(by: i * n), 1, y.advanced(by: i), vDSP_Length(n))
                        }
                    }
                    count = m
                }
            )
        }
    }

    public func threshold(to lowerBound: Matrix, with rule: vDSP.ThresholdRule<Float>) -> Matrix {
        let bRow = lowerBound.shape.rows == 1 && lowerBound.shape.columns == shape.columns
        let bCol = lowerBound.shape.rows == shape.rows && lowerBound.shape.columns == 1
        precondition(bRow || bCol)
        let m = shape.rows
        let n = shape.columns
        let f =
            switch rule {
            case .clampToThreshold:
                vDSP_vthr
            case .signedConstant(let replacement):
                {
                    var C = replacement
                    vDSP_vthrsc($0, $1, $2, &C, $3, $4, $5)
                }
            case .zeroFill:
                vDSP_vthres
            @unknown default:
                vDSP_vthr
            }
        return .init(
            shape: shape,
            elements: [Scalar](unsafeUninitializedCapacity: count) {
                yptr, initializedNum in
                let y = yptr.baseAddress!
                elements.withUnsafeBufferPointer {
                    let x = $0.baseAddress!
                    lowerBound.elements.withUnsafeBufferPointer {
                        let th = $0.baseAddress!
                        if bRow {
                            for i in 0..<n {
                                f(
                                    x.advanced(by: i), n,
                                    th.advanced(by: i),
                                    y.advanced(by: i), n,
                                    vDSP_Length(m))
                            }
                        } else {
                            for i in 0..<m {
                                f(
                                    x.advanced(by: i * n), 1,
                                    th.advanced(by: i),
                                    y.advanced(by: i * n), 1,
                                    vDSP_Length(n))
                            }
                        }
                    }
                }
                initializedNum = count
            }
        )
    }

    public mutating func thresholdInplace(
        to lowerBound: Matrix, with rule: vDSP.ThresholdRule<Float>
    ) {
        let bRow = lowerBound.shape.rows == 1 && lowerBound.shape.columns == shape.columns
        let bCol = lowerBound.shape.rows == shape.rows && lowerBound.shape.columns == 1
        precondition(bRow || bCol)
        let m = shape.rows
        let n = shape.columns
        let f =
            switch rule {
            case .clampToThreshold:
                vDSP_vthr
            case .signedConstant(let replacement):
                {
                    var C = replacement
                    vDSP_vthrsc($0, $1, $2, &C, $3, $4, $5)
                }
            case .zeroFill:
                vDSP_vthres
            @unknown default:
                vDSP_vthr
            }
        elements.withUnsafeMutableBufferPointer {
            let x = $0.baseAddress!
            lowerBound.elements.withUnsafeBufferPointer {
                let th = $0.baseAddress!
                if bRow {
                    for i in 0..<n {
                        f(
                            x.advanced(by: i), n,
                            th.advanced(by: i),
                            x.advanced(by: i), n,
                            vDSP_Length(m))
                    }
                } else {
                    for i in 0..<m {
                        f(
                            x.advanced(by: i * n), 1,
                            th.advanced(by: i),
                            x.advanced(by: i * n), 1,
                            vDSP_Length(n))
                    }
                }
            }
        }
    }

    public mutating func thresholdInplace(
        to lowerBound: Scalar, with rule: vDSP.ThresholdRule<Float>
    ) {
        fmapInplace { vDSP.threshold($0, to: lowerBound, with: rule, result: &$0) }
    }
}
