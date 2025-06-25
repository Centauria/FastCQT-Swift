import Accelerate
import Foundation
import Plinth

public func parabolicInterpolation(_ x: [Float]) -> [Float] {
    let n = x.count
    precondition(n >= 3, "Input length must be at least 3")

    let a = vDSP.add(multiplication: (a: x[1..<n - 1], b: -2), vDSP.add(x[2..<n], x[0..<n - 2]))
    let b = vDSP.divide(vDSP.subtract(x[0..<n - 2], x[2..<n]), 2)
    let a_ = vDSP.absolute(a)
    let b_ = vDSP.absolute(b)

    let shifts = [Float](unsafeUninitializedCapacity: n) { ptr, count in
        ptr[0] = 0
        ptr[n - 1] = 0
        for i in 0..<n - 2 {
            ptr[i + 1] = b_[i] > a_[i] ? 0 : b[i] / a[i]
        }
        count = n
    }

    return shifts
}

public func parabolicInterpolation(_ x: Matrix<Float>) -> Matrix<Float> {
    let m = x.shape.columns
    let n = x.shape.rows
    precondition(n >= 3, "Input matrix rows must be at least 3")
    let a = x[2..<n, 0..<m] + x[0..<n - 2, 0..<m] - 2.0 * x[1..<n - 1, 0..<m]
    let b = (x[0..<n - 2, 0..<m] - x[2..<n, 0..<m]) / 2.0
    let a_ = a.absolute()
    let b_ = b.absolute()

    var shifts: Matrix<Float> = .zeros(shape: x.shape)
    shifts[1..<n - 1, 0..<m] =
        (a_[0..<n - 2, 0..<m] - b_[0..<n - 2, 0..<m] >= 0)
        * b[0..<n - 2, 0..<m]
        / a[0..<n - 2, 0..<m]

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
    let fftFreqs = fftFrequencies(sr: sr, nFFT: nFFT)
    let avg = gradient(y: S)
}
