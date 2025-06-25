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

public func resample(
    x: [Float], inSampleRate: Double, outSampleRate: Double, scale: Bool
) -> [Float] {
    let resampleFactor = outSampleRate / inSampleRate
    let inum = x.count
    let onum = Int(ceil(Double(inum) * resampleFactor))
    var q_spec = soxr_quality_spec(UInt(SOXR_HQ), 0)
    var idone = 0
    var odone = 0

    var output = [Float](unsafeUninitializedCapacity: onum) { y, count in
        let yptr = y.baseAddress!
        x.withUnsafeBufferPointer {
            let xptr = $0.baseAddress!
            _ = soxr_oneshot(
                inSampleRate, outSampleRate, 1,
                xptr, inum, &idone,
                yptr, onum, &odone,
                nil, &q_spec, nil)
        }
        vDSP_vclr(yptr.advanced(by: odone), 1, vDSP_Length(onum - odone))
        count = onum
    }
    if scale {
        let factor = Float(resampleFactor).squareRoot()
        vDSP.divide(output, factor, result: &output)
    }
    return output
}

public func downsample(x: [Float], downsampleFactor: Int, scale: Bool) -> [Float] {
    resample(x: x, inSampleRate: Double(downsampleFactor), outSampleRate: 1, scale: scale)
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
