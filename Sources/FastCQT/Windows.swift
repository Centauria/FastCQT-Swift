import Accelerate
import Foundation

public struct Windows {
    public enum WindowType {
        case ones
        case hann
    }

    public static let Bandwidth: [WindowType: Float] = [
        .ones: 1.0,
        .hann: 1.50018310546875,
    ]

    public static func get(type: WindowType, M: Int) -> [Float] {
        switch type {
        case .ones:
            [Float](repeating: 1.0, count: M)
        case .hann:
            generalHamming(M: M, alpha: 0.5)
        }
    }

    private static func generalCosine(M: Int, a: [Float]) -> [Float] {
        let fac = vDSP.ramp(in: -Float.pi...Float.pi, count: M + 1)[0..<M]
        var w = [Float](repeating: 0, count: M)
        for k in 0..<a.count {
            let c = vDSP.multiply(a[k], vForce.cos(vDSP.multiply(Float(k), fac)))
            vDSP.add(w, c, result: &w)
        }
        return w
    }

    private static func generalHamming(M: Int, alpha: Float) -> [Float] {
        generalCosine(M: M, a: [alpha, 1 - alpha])
    }
}
