import Accelerate
import Foundation

public func parabolicInterpolation(_ x: [Float]) -> [Float] {
    let n = x.count
    precondition(n >= 3, "Input length must be at least 3")

    let a = vDSP.add(multiplication: (a: x[1..<n - 1], b: -2), vDSP.add(x[2..<n], x[0..<n - 2]))
    let b = vDSP.divide(vDSP.subtract(x[0..<n - 2], x[2..<n]), 2)
    let a_ = vDSP.absolute(a)
    let b_ = vDSP.absolute(b)

    let shifts = [Float](unsafeUninitializedCapacity: n) { ptr, count in
        ptr[0] = ptr[n - 1] = 0
        for i in 0..<n - 2 {
            ptr[i + 1] = b_[i] > a_[i] ? 0 : b[i] / a[i]
        }
        count = n
    }

    return shifts
}

public func piptrack(
    y: [Float],
    sr: Float = 22050,
    nFFT: Int = 2048,
    hopLength: Int? = nil,
    fmin: Float = 150.0,
    fmax: Float = 4000.0,
    threshold: Float = 0.1,
    window: Windows.WindowType = .hann,
    center: Bool = true
) {
    let S = spectrogram(y: y, nFFT: nFFT, hopLength: hopLength, window: window, center: center)
    let fMin = max(fmin, 0)
    let fMax = min(fmax, sr / 2)
    
}
