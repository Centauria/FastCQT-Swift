import Accelerate
import Foundation
import Soxr

public func earlyDownsampleCount(
    nyquist: Float, filterCutoff: Float, hopLength: Int, nOctaves: Int
) -> Int {
    let downsampleCount1 = max(0, Int(ceilf(log2f(nyquist / filterCutoff)) - 1) - 1)
    let numTwos = numTwoFactors(hopLength)
    let downsampleCount2 = max(0, numTwos - nOctaves + 1)
    return min(downsampleCount1, downsampleCount2)
}

public func downsample(x: [Float], downsampleFactor: Int, scale: Bool) -> [Float] {
    let inum = x.count
    let onum = divceil(inum, downsampleFactor)
    var output = [Float](repeating: 0, count: onum)
    var q_spec = soxr_quality_spec(UInt(SOXR_HQ), 0)
    var idone = 0
    var odone = 0

    x.withUnsafeBufferPointer { xptr in
        output.withUnsafeMutableBufferPointer { yptr in
            _ = soxr_oneshot(
                Double(downsampleFactor), 1.0, 1,
                xptr.baseAddress, inum, &idone,
                yptr.baseAddress, onum, &odone,
                nil, &q_spec, nil)
        }
    }
    while odone < onum {
        output[odone] = 0
        odone += 1
    }
    if scale {
        let factor = sqrt(Float(downsampleFactor))
        vDSP.multiply(factor, output, result: &output)
    }
    return output
}

public func earlyDownsample(
    y: [Float], sr: Float, hopLength: Int, nOctaves: Int, nyquist: Float, filterCutoff: Float,
    scale: Bool
) -> ([Float], Float, Int) {
    let downsampleCount = earlyDownsampleCount(
        nyquist: nyquist, filterCutoff: filterCutoff, hopLength: hopLength, nOctaves: nOctaves)
    guard downsampleCount > 0 else {
        return (y, sr, hopLength)
    }
    let downsampleFactor = 1 << downsampleCount
    let newHopLength = hopLength >> downsampleCount
    assert(y.count >= downsampleFactor)
    let newSr = sr / Float(downsampleFactor)
    let output = downsample(x: y, downsampleFactor: downsampleFactor, scale: true)
    return (output, newSr, newHopLength)
}
