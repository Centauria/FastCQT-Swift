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
    norm: Float?,
    sparsity: Float,
    hopLength: Int?,
    window: Windows.WindowType,
    gamma: Float? = 0,
    alpha: [Float]?
) -> SparseMatrix_ComplexFloat {
    var (basis, lengths) = wavelet(
        freqs: freqs, sr: sr, window: window,
        filterScale: filterScale, norm: norm,
        gamma: gamma, alpha: alpha)
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

public func trimStack(cqtResponse: [Matrix<Float>], nBins: Int) -> Matrix<Float> {
    precondition(cqtResponse.count > 0)
    let maxRow = cqtResponse.map { $0.shape.rows }.min()!
    var cqtOut: Matrix<Float> = .zeros(shape: .init(rows: maxRow, columns: nBins))

    var end = nBins
    let all = 0..<maxRow
    for ci in cqtResponse {
        let nOct = ci.shape.columns
        if end < nOct {
            cqtOut[all, 0..<end] = ci[all, nOct - end..<nOct]
        } else {
            cqtOut[all, end - nOct..<end] = ci[all, 0..<nOct]
        }
        end -= nOct
    }
    return cqtOut
}

public func trimStack(cqtResponse: [ComplexMatrix<Float>], nBins: Int) -> ComplexMatrix<Float> {
    precondition(cqtResponse.count > 0)
    let maxRow = cqtResponse.map { $0.shape.rows }.min()!
    var cqtOut: ComplexMatrix<Float> = .zeros(shape: .init(rows: maxRow, columns: nBins))

    var end = nBins
    let all = 0..<maxRow
    for ci in cqtResponse {
        let nOct = ci.shape.columns
        if end < nOct {
            cqtOut[all, 0..<end] = ci[all, nOct - end..<nOct]
        } else {
            cqtOut[all, end - nOct..<end] = ci[all, 0..<nOct]
        }
        end -= nOct
    }
    return cqtOut
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
    norm: Float? = 1,
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

public func VQT(
    y: [Float],
    sr: Float = 22050,
    hopLength: Int = 512,
    fmin: Float = 32.70319566257483,
    nBins: Int = 84,
    gamma: Float? = nil,
    binsPerOctave: Int = 12,
    tuning: Float? = 0,
    filterScale: Float = 1,
    norm: Float? = 1,
    sparsity: Float = 0.01,
    window: Windows.WindowType = .hann,
    scale: Bool = true
) -> ComplexMatrix<Float> {
    let nOctaves = divceil(nBins, binsPerOctave)
    let nFilters = min(binsPerOctave, nBins)
    let tune = tuning ?? estimateTuning(y: y, sr: sr, binsPerOctave: binsPerOctave)
    let freqs = cqtFrequencies(nBins: nBins, fMin: fmin, binsPerOctave: binsPerOctave, tuning: tune)
    let alpha =
        nBins == 1
        ? etRelativeBW(binsPerOctave: binsPerOctave)
        : relativeBandwidth(freqs: freqs)
    let (_, filterCutoff) = waveletLengths(
        freqs: freqs, sr: sr,
        window: window, filterScale: filterScale,
        gamma: gamma, alpha: alpha)
    let nyquist = sr / 2
    precondition(filterCutoff <= nyquist)

    var (y2, sr2, hop2) = earlyDownsample(
        y: y, sr: sr, hopLength: hopLength,
        nOctaves: nOctaves, nyquist: nyquist,
        filterCutoff: filterCutoff, scale: scale)
    let origSr = sr2
    var factor: Float = 1
    var vqtResp: [ComplexMatrix<Float>] = []
    for i in 0..<nOctaves {
        let s = max(0, nBins - nFilters * (i + 1))
        let e = nBins - nFilters * i
        let sl = s..<e
        let freqsOct = Array(freqs[sl])
        let alphaOct = Array(alpha[sl])

        var fftBasis = vqtFilterFFT(
            sr: sr2, freqs: freqsOct,
            filterScale: filterScale,
            norm: norm, sparsity: sparsity,
            hopLength: 0, window: .hann,
            gamma: gamma, alpha: alphaOct)
        let nFFT = Int(2 * (fftBasis.real.structure.rowCount - 1))
        fftBasis *= factor.squareRoot()

        vqtResp.append(cqtResponse(y: y2, nFFT: nFFT, hopLength: hop2, fftBasis: fftBasis))

        if hop2 % 2 == 0 {
            hop2 /= 2
            sr2 /= 2
            y2 = resample(x: y2, inSampleRate: 2, outSampleRate: 1, scale: true)
        }

        factor *= 2
    }

    var V = trimStack(cqtResponse: vqtResp, nBins: nBins)

    if scale {
        let numFrames = V.shape.rows
        var (lengths, _) = waveletLengths(
            freqs: freqs, sr: origSr,
            window: window, filterScale: filterScale,
            gamma: gamma, alpha: alpha)
        vForce.sqrt(lengths, result: &lengths)
        for i in 0..<nBins {
            V[0..<numFrames, i] /= lengths[i]
        }
    }
    return V
}

public func CQT(
    y: [Float],
    sr: Float = 22050,
    hopLength: Int = 512,
    fmin: Float = 32.70319566257483,
    nBins: Int = 84,
    binsPerOctave: Int = 12,
    tuning: Float? = 0,
    filterScale: Float = 1,
    norm: Float? = 1,
    sparsity: Float = 0.01,
    window: Windows.WindowType = .hann,
    scale: Bool = true
) -> ComplexMatrix<Float> {
    VQT(
        y: y, sr: sr, hopLength: hopLength,
        fmin: fmin, nBins: nBins, gamma: 0,
        binsPerOctave: binsPerOctave, tuning: tuning,
        filterScale: filterScale, norm: norm,
        sparsity: sparsity, window: window, scale: scale)
}

public func hybridCQT(
    y: [Float],
    sr: Float = 22050,
    hopLength: Int = 512,
    fmin: Float = 32.70319566257483,
    nBins: Int = 84,
    binsPerOctave: Int = 12,
    tuning: Float? = 0,
    filterScale: Float = 1,
    norm: Float? = 1,
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

    let (lengths, _) = waveletLengths(
        freqs: freqs, sr: sr, window: window,
        filterScale: filterScale, alpha: alpha)

    let normLengths = vForce.exp2(vForce.ceil(vForce.log2(lengths)))
    let indexPseudo = normLengths.firstIndex { $0 < Float(2 * hopLength) } ?? nBins
    var cqtResponse: [Matrix<Float>] = []
    if indexPseudo < nBins {
        cqtResponse.append(
            pseudoCQT(
                y: y, sr: sr,
                hopLength: hopLength, fmin: freqs[indexPseudo],
                nBins: nBins - indexPseudo,
                binsPerOctave: binsPerOctave, filterScale: filterScale,
                norm: norm, sparsity: sparsity,
                window: window, scale: scale)
        )
    }
    if indexPseudo > 0 {
        cqtResponse.append(
            CQT(
                y: y, sr: sr, hopLength: hopLength, fmin: freqs[0], nBins: indexPseudo,
                binsPerOctave: binsPerOctave, filterScale: filterScale, norm: norm,
                sparsity: sparsity, window: window, scale: scale
            ).absolute()
        )
    }

    return trimStack(cqtResponse: cqtResponse, nBins: nBins)
}
