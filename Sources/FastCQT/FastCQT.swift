// The Swift Programming Language
// https://docs.swift.org/swift-book

import Accelerate
import CoreML
import Foundation
import Numerics
import Plinth

public func fftFrequencies(sr: Float = 22050, nFFT: Int = 2048) -> [Float] {
    vDSP.ramp(in: 0...sr / 2, count: 1 + nFFT / 2)
}

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

public func vqtFilterFFT(
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
        case let l = exp2(1 + ceilf(log2f(Float(hop)))),
        Float(nFFT) < l
    {
        nFFT = Int(l)
    }

    let lengthsN: Matrix<Float> = .init(row: lengths) / Float(nFFT)
    for i in 0..<initFFT {
        basis[0..<n, i] *= lengthsN
    }

    let plan = FFT<Float>.createSetup(shape: .row(length: nFFT))
    defer {
        FFT<Float>.destroySetup(plan)
    }
    var basisExtended: ComplexMatrix<Float> = .zeros(shape: .init(rows: n, columns: nFFT))
    basisExtended[0..<n, 0..<initFFT] = basis

    for i in 0..<n {
        basisExtended[i, 0..<nFFT] = basisExtended[i, 0..<nFFT].fft1D(setup: plan)
    }

    let fftBasis = basisExtended[0..<n, 0...nFFT / 2]
    let fftBasisSparse = sparsify_rows(x: fftBasis, quantile: sparsity)

    return fftBasisSparse
}

public func cqtResponse(
    y: [Float],
    nFFT: Int,
    hopLength: Int,
    fftBasis: SparseMatrix_ComplexFloat,
    window: Windows.WindowType = .ones
) -> ComplexMatrix<Float> {
    let D = stft(signal: y, nFFT: nFFT, hopLength: hopLength, window: window)
    let outputFlat = SparseMultiply(D, fftBasis)
    return outputFlat
}

public func cqtResponse(
    y: [Float],
    nFFT: Int,
    hopLength: Int,
    fftBasis: SparseMatrix_Float,
    window: Windows.WindowType = .ones
) -> Matrix<Float> {
    let D = stft(signal: y, nFFT: nFFT, hopLength: hopLength, window: window)
    let Dmag = D.absolute()
    let outputFlat = SparseMultiply(Dmag, fftBasis)
    return outputFlat
}

public func pseudoCQT(
    y: [Float],
    sr: Float = 22050,
    hopLength: Int = 512,
    fmin: Float = 32.70319566257483,
    nBins: Int = 84,
    binsPerOctave: Int = 12,
    tuning: Float? = 0,
    filterScale: Float = 1,
    norm: Float = 1,
    sparsity: Float = 0.01,
    window: Windows.WindowType = .hann,
    scale: Bool = true
) -> Matrix<Float> {
    let tune = tuning ?? estimateTuning(y: y, sr: sr, binsPerOctave: binsPerOctave)
    let freqs = cqtFrequencies(nBins: nBins, fMin: fmin, binsPerOctave: binsPerOctave, tuning: tune)

    let alpha =
        nBins == 1
        ? etRelativeBW(binsPerOctave: binsPerOctave)
        : relativeBandwidth(freqs: freqs)

    let fftBasis = vqtFilterFFT(
        sr: sr, freqs: freqs, filterScale: filterScale,
        norm: norm, sparsity: sparsity,
        hopLength: hopLength, window: window, alpha: alpha)
    let nFFT = Int(2 * (fftBasis.real.structure.rowCount - 1))

    let fftBasisMag = fftBasis.absolute()
    defer {
        SparseCleanup(fftBasisMag)
    }
    var output = cqtResponse(
        y: y, nFFT: nFFT, hopLength: hopLength,
        fftBasis: fftBasisMag, window: .hann)

    if scale {
        output /= sqrtf(Float(nFFT))
    } else {
        let numFrames = output.shape.rows
        var (lengths, _) = waveletLengths(freqs: freqs, sr: sr, window: window, alpha: alpha)
        vDSP.divide(lengths, Float(nFFT), result: &lengths)
        vForce.sqrt(lengths, result: &lengths)
        for i in 0..<nBins {
            output[0..<numFrames, i] *= lengths[i]
        }
    }

    return output
}
