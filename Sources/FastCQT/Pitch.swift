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

public func estimateTuning(
    y: [Float],
    sr: Float = 22050,
    nFFT: Int = 2048,
    resolution: Float = 0.01,
    binsPerOctave: Int = 12
) -> Float {
    let (pitch, mag) = piptrack(y: y, sr: sr, nFFT: nFFT)
    let threshold: Float = mag.median() ?? 0
    let filteredPitch = zip(pitch, mag).compactMap { (p, m) in
        m >= threshold ? p : nil
    }
    return pitchTuning(
        frequencies: filteredPitch, resolution: resolution, binsPerOctave: binsPerOctave)
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
    center: Bool = true,
    ref: Float? = nil
) -> ([Float], [Float]) {
    var S = spectrogram(y: y, nFFT: nFFT, hopLength: hopLength, window: window, center: center)
    let row = S.shape.rows
    let fMin = max(fmin, 0)
    let fMax = min(fmax, sr / 2)
    let fDelta = sr / Float(nFFT)
    let fMinIndex = Int(fMin / fDelta) + (remainder(fMin, fDelta) == 0 ? 0 : 1)
    let fMaxIndex = Int(fMax / fDelta) - (remainder(fMax, fDelta) == 0 ? 1 : 0)

    /// These preconditions restrict certain extreme inputs, but still cover the situations used in this project
    precondition(fMinIndex > 0)
    precondition(fMaxIndex < nFFT / 2)

    let avg = gradient(y: S, axis: 1)
    let shift = parabolicInterpolation(S, axis: 1)
    let dskew = 0.5 * avg * shift

    if let r = ref {
        S.thresholdInplace(to: abs(r), with: .zeroFill)
    } else {
        let reference = threshold * S.maximum(axis: 1)
        S.thresholdInplace(to: reference, with: .zeroFill)
    }
    var pitches: [Float] = []
    var mags: [Float] = []
    let appendElement = { (j: Int, i: Int) in
        pitches.append((Float(i) + shift[j, i]) * fDelta)
        mags.append(S[j, i] + dskew[j, i])
    }
    for j in 0..<row {
        for i in fMinIndex...fMaxIndex {
            if S[j, i] > S[j, i - 1] && S[j, i] >= S[j, i + 1] {
                appendElement(j, i)
            }
        }
    }
    return (pitches, mags)
}

public func pitchTuning(
    frequencies: [Float],
    resolution: Float = 0.01,
    binsPerOctave: Int = 12
) -> Float {
    let positiveFreqs = frequencies.filter { $0 > 0 }
    guard positiveFreqs.count > 0 else { return 0 }

    let octs = vDSP.multiply(Float(binsPerOctave), HzToOcts(frequencies: positiveFreqs))
    let residual = octs.map {
        let r = remainder($0, 1)
        return r >= 0.5 ? r - 1 : r
    }

    let bins: [Float] = vDSP.ramp(in: -0.5...0.5, count: Int(ceil(1.0 / resolution)) + 1)
    let counts = residual.histogram(bins: bins)
    let tuningEst = bins[counts.argmax!]

    return tuningEst
}
