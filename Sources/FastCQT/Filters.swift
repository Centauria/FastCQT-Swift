import Accelerate
import Foundation
import Numerics
import Plinth

public func windowBandwidth(window: [Float]) -> Float {
    let n = window.count
    let sw = vDSP.sum(window)
    let sw2 = vDSP.sumOfSquares(window)
    let y = Float(n) * sw2 / (sw * sw + .ulpOfOne)
    return y
}

public func etRelativeBW(binsPerOctave: Int) -> [Float] {
    let r = exp2(1.0 / Float(binsPerOctave))
    return [r]
}

public func relativeBandwidth(freqs: [Float]) -> [Float] {
    let n = freqs.count
    let logf = vForce.log2(freqs)
    var bpo = [Float](repeating: 0, count: n)
    bpo[0] = 1 / (logf[1] - logf[0])
    if n > 1 {
        bpo[n - 1] = 1 / (logf[n - 1] - logf[n - 2])
    }
    if n > 2 {
        let middleCount = n - 2

        logf.withUnsafeBufferPointer { logPtr in
            bpo.withUnsafeMutableBufferPointer { bpoPtr in
                // 直接在bpo的中间部分计算差值，然后求倒数
                // 先计算差值存到 bpo[1...n-2]
                vDSP_vsub(
                    logPtr.baseAddress!,  // logf[0...n-3]
                    1,
                    logPtr.baseAddress! + 2,  // logf[2...n-1]
                    1,
                    bpoPtr.baseAddress! + 1,  // 暂存到 bpo[1...n-2]
                    1,
                    vDSP_Length(middleCount)
                )

                // 然后计算 2 / bpo[1...n-2]，结果还是存回 bpo[1...n-2]
                var two: Float = 2.0
                vDSP_svdiv(
                    &two,
                    bpoPtr.baseAddress! + 1,  // 从 bpo[1] 读取
                    1,
                    bpoPtr.baseAddress! + 1,  // 写回到 bpo[1]
                    1,
                    vDSP_Length(middleCount)
                )
            }
        }
    }
    let u = vForce.exp2(vDSP.divide(2.0, bpo))
    let alpha = vDSP.divide(vDSP.add(-1.0, u), vDSP.add(1.0, u))
    return alpha
}

public func waveletLengths(
    freqs: [Float], sr: Float, window: Windows.WindowType,
    filterScale: Float = 1, gamma: Float? = 0, alpha: [Float]?
) -> ([Float], Float) {
    guard filterScale > 0 else { return ([], .nan) }
    let alpha_ = alpha ?? relativeBandwidth(freqs: freqs)
    let bw = Windows.Bandwidth[window]! / 2
    let a = vDSP.add(multiplication: (a: alpha_, b: bw / filterScale), 1)
    let fCutoff: Float
    let lengths: [Float]
    if let gamma_ = gamma {
        fCutoff = vDSP.maximum(vDSP.add(multiplication: (a: a, b: freqs), gamma_ / 2))
        lengths = vDSP.divide(
            filterScale * sr,
            vDSP.add(multiplication: (a: alpha_, b: freqs), gamma_))
    } else {
        let gamma_ = vDSP.multiply(24.7 / 0.108, alpha_)
        fCutoff = vDSP.maximum(
            vDSP.add(multiplication: (a: a, b: freqs), vDSP.divide(gamma_, 2)))
        lengths = vDSP.divide(
            filterScale * sr,
            vDSP.add(multiplication: (a: alpha_, b: freqs), gamma_))
    }
    return (lengths, fCutoff)
}

public func wavelet(
    freqs: [Float],
    sr: Float = 22050,
    window: Windows.WindowType = .hann,
    filterScale: Float = 1,
    padFFT: Bool = true,
    norm: Float = 1,
    gamma: Float = 0,
    alpha: [Float]?
) -> (ComplexMatrix<Float>, [Float]) {
    let (lengths, _) = waveletLengths(freqs: freqs, sr: sr, window: window, alpha: alpha)
    let n = freqs.count

    let maxlenFloat = vDSP.maximum(lengths)
    let maxlen: size_t
    if padFFT {
        maxlen = 1 << Int(ceilf(log2f(maxlenFloat)))
    } else {
        maxlen = size_t(ceilf(maxlenFloat))
    }

    var filters: ComplexMatrix<Float> = .zeros(shape: .init(rows: n, columns: maxlen))

    for i in 0..<n {
        let ilen = lengths[i]
        let freq = freqs[i]
        var angles = vDSP.ramp(
            withInitialValue: floorf(-ilen / 2), increment: 1,
            count: Int(floorf(ilen / 2) - floorf(-ilen / 2)))
        vDSP.multiply(2 * .pi * freq / sr, angles, result: &angles)
        var sig = phasor(angles: angles)
        let siglen = sig.count
        let win: Matrix<Float> = .init(column: Windows.get(type: window, M: siglen))
        sig *= win
        sig = normalize(S: sig, norm: norm)
        let start = (maxlen - siglen) / 2
        filters[i, start..<start + siglen] = sig
    }
    return (filters, lengths)
}
