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
    var values: [Complex<Float>] = []
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
                values.append(xi[i, k])
                rowIndices.append(Int32(k))
                columnIndices.append(Int32(i))
            }
        }
    }
    let (reals, imags) = values.withUnsafeBufferPointer { buf in
        let count = buf.count
        let reals: [Float] = [Float](unsafeUninitializedCapacity: count) { ptr, initializedCount in
            for i in 0..<count {
                ptr[i] = buf[i].real
            }
            initializedCount = count
        }
        let imags: [Float] = [Float](unsafeUninitializedCapacity: count) { ptr, initializedCount in
            for i in 0..<count {
                ptr[i] = buf[i].imaginary
            }
            initializedCount = count
        }
        return (reals, imags)
    }
    return .init(
        _real: SparseConvertFromCoordinate(
            Int32(ncols), Int32(nrows),
            values.count, UInt8(1), .init(),
            &rowIndices, &columnIndices, reals),
        _imag: imags
    )
}
