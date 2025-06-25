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

public func parabolicInterpolation(_ x: Matrix<Float>, axis: Int = 0) -> Matrix<Float> {
    precondition((0...1).contains(axis), "axis must be 0(row) or 1(column)")
    let m = x.shape.columns
    let n = x.shape.rows
    precondition(n >= 3, "Input matrix rows must be at least 3")

    var shifts: Matrix<Float> = .zeros(shape: x.shape)
    if axis == 0 {
        let all = 0..<m
        let head = 0..<n - 2
        let mid = 1..<n - 1
        let tail = 2..<n
        let a = x[tail, all] + x[head, all] - 2.0 * x[mid, all]
        let b = (x[head, all] - x[tail, all]) / 2.0
        let a_ = a.absolute()
        let b_ = b.absolute()
        shifts[mid, all] =
            (a_[head, all] - b_[head, all] >= 0)
            * b[head, all] / a[head, all]
    } else if axis == 1 {
        let all = 0..<n
        let head = 0..<m - 2
        let mid = 1..<m - 1
        let tail = 2..<m
        let a = x[all, tail] + x[all, head] - 2.0 * x[all, mid]
        let b = (x[all, head] - x[all, tail]) / 2.0
        let a_ = a.absolute()
        let b_ = b.absolute()
        shifts[all, mid] =
            (a_[all, head] - b_[all, head] >= 0)
            * b[all, head] / a[all, head]
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
    let fftFreqs = fftFrequencies(sr: sr, nFFT: nFFT)
    let avg = gradient(y: S, axis: 1)
    let c = 1
}
