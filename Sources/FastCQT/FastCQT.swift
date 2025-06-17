// The Swift Programming Language
// https://docs.swift.org/swift-book

import Accelerate
import CoreML
import Foundation
import Numerics
import Plinth

public func cqtFrequencies(
    nBins: Int, fMin: Float, binsPerOctave: Int = 12, tuning: Float = 0.0
)
    -> [Float]
{
    var freqs: [Float] = vDSP.ramp(withInitialValue: tuning, increment: 1, count: nBins)
    vDSP.divide(freqs, Float(binsPerOctave), result: &freqs)
    vForce.exp2(freqs, result: &freqs)
    vDSP.multiply(fMin, freqs, result: &freqs)
    return freqs
}

public func VQTFilterFFT(
    sr: Float,
    freqs: [Float],
    filterScale: Float,
    norm: Float,
    sparsity: Float,
    hopLength: Int?,
    window: Windows.WindowType,
    gamma: Float = 0,
    alpha: [Float]?
) -> SparseMatrix_ComplexFloat {
    var (basis, lengths) = wavelet(
        freqs: freqs, sr: sr, window: window,
        filterScale: filterScale, norm: norm, gamma: gamma,
        alpha: alpha)
    let n = basis.shape.rows
    let initFFT = basis.shape.columns
    var nFFT = basis.shape.columns

    if let hop = hopLength,
        case let l = powf(2.0, 1 + ceilf(log2f(Float(hop)))),
        Float(nFFT) < l
    {
        nFFT = Int(l)
    }

    let lengthsN: Matrix<Float> = .init(row: lengths) / Float(nFFT)
    for i in 0..<initFFT {
        basis[0..<n, i] *= lengthsN
    }

    let plan = FFT<Float>.createSetup(shape: .row(length: nFFT))
    var basisExtended: ComplexMatrix<Float> = .zeros(shape: .init(rows: n, columns: nFFT))
    basisExtended[0..<n, 0..<initFFT] = basis

    for i in 0..<n {
        basisExtended[i, 0..<nFFT] = basisExtended[i, 0..<nFFT].fft1D(
            direction: .forward, setup: plan)
    }

    let fftBasis = basisExtended[0..<n, 0...nFFT / 2]
    let fftBasisSparse = sparsify_rows(x: fftBasis, quantile: sparsity)

    return fftBasisSparse
}
