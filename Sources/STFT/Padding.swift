import Accelerate
import Foundation

public enum PaddingType {
    case zeros
    case reflect
}

public func pad(x: [Float], padSize: (Int, Int), type: PaddingType) -> [Float] {
    let (padL, padR) = padSize
    let n = x.count + padL + padR
    if type == .reflect {
        precondition(
            padL < x.count && padR < x.count,
            "Padding length must be shorter than input length when in reflect padding mode"
        )
    }
    let y = [Float](unsafeUninitializedCapacity: n) { b, initializedCount in
        let yp = b.baseAddress!
        x.withUnsafeBufferPointer {
            let xp = $0.baseAddress!
            vDSP_mmov(xp, yp.advanced(by: padL), 1, vDSP_Length(x.count), 1, 1)
        }
        switch type {
        case .zeros:
            vDSP_vclr(yp, 1, vDSP_Length(padL))
            vDSP_vclr(yp.advanced(by: x.count + padL), 1, vDSP_Length(padR))
        case .reflect:
            for i in 0..<padL {
                yp[i] = x[padL - i]
            }
            for i in 0..<padR {
                yp[x.count + padL + i] = x[x.count - 2 - i]
            }
        }
        initializedCount = n
    }
    return y
}
