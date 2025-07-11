import Accelerate
import Foundation
import Numerics
import Plinth

func divceil(_ a: Int, _ b: Int) -> Int {
    let d = div(Int32(a), Int32(b))
    return Int(d.quot + (d.rem > 0 ? 1 : 0))
}

@inline(__always)
func numTwoFactors(_ x: Int) -> Int {
    guard x > 0 else { return 0 }
    return x.trailingZeroBitCount
}

func phasor(angles: [Float]) -> ComplexMatrix<Float> {
    let n = angles.count
    var u = [Float](repeating: 0, count: n)
    var v = [Float](repeating: 0, count: n)
    vForce.sincos(angles, sinResult: &v, cosResult: &u)
    let z: ComplexMatrix<Float> = .init(
        real: .init(shape: .row(length: n), elements: u),
        imaginary: .init(shape: .row(length: n), elements: v))
    return z
}

func phasor2(angles: [Float32]) -> [Complex<Float32>] {
    return angles.map { a in
        Complex.exp(.init(0, a))
    }
}

func normalize(S: ComplexMatrix<Float>, norm: Float = 1) -> ComplexMatrix<Float> {
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

func sparsify_rows(
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

func gradient(y: Matrix<Float>, axis: Int = 0) -> Matrix<Float> {
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

func HzToOcts(frequencies: [Float], tuning: Float = 0, binsPerOctave: Int = 12) -> [Float] {
    let A440 = 440.0 * exp2(tuning / Float(binsPerOctave))
    return vForce.log2(vDSP.divide(frequencies, A440 / 16.0))
}
