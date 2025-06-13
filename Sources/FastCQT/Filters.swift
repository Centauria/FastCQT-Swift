import Accelerate
import Foundation

public func windowBandwidth(window: [Float]) -> Float {
    let n = window.count
    let sw = vDSP.sum(window)
    let sw2 = vDSP.sumOfSquares(window)
    let y = Float(n) * sw2 / (sw * sw + FLT_EPSILON)
    return y
}

public func etRelativeBW(binsPerOctave: Int) -> Float {
    let r = powf(2.0, 1.0 / Float(binsPerOctave))
    return r
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
    let u = vForce.exp(vDSP.multiply(log(2.0), vDSP.divide(2.0, bpo)))
    let alpha = vDSP.divide(vDSP.add(-1.0, u), vDSP.add(1.0, u))
    return alpha
}
