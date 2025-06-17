import Accelerate
import Foundation
import Numerics
import Plinth

public func stft(
    signal: [Float],
    nFFT: Int,
    hopLength: Int,
    window: Windows.WindowType = .hann,
    center: Bool = true,
    normalized: Bool = true
) -> ComplexMatrix<Float> {
    var inputSignal: Matrix<Float>
    if center {
        let n = signal.count
        let startK = divceil(nFFT / 2, hopLength)
        let tailK = (n + nFFT / 2 - nFFT) / hopLength + 1
        let padL = nFFT / 2
        let padR = (tailK <= startK || tailK * hopLength - nFFT / 2 * 2 + nFFT <= n) ? nFFT / 2 : 0
        let input = [Float](repeating: 0, count: padL) + signal + [Float](repeating: 0, count: padR)
        inputSignal = .init(shape: .row(length: n + padL + padR), elements: input)
    } else {
        inputSignal = .init(shape: .row(length: signal.count), elements: signal)
    }

    let numSamples = inputSignal.count
    let lastFrameComplete = (numSamples - nFFT) % hopLength == 0
    let numFrames = (numSamples - nFFT) / hopLength + 1

    let plan = Plinth.FFT<Float>.createSetup(shape: .row(length: nFFT))
    var result: ComplexMatrix<Float> = .zeros(shape: .init(rows: numFrames, columns: nFFT / 2 + 1))

    for i in 0..<numFrames {
        let length = lastFrameComplete || i == numFrames ? numSamples - i * hopLength : nFFT
        var input: Matrix<Float> = .zeros(shape: .row(length: nFFT))
        input[0, 0..<length] = inputSignal[0, i * hopLength..<i * hopLength + length]
        result[i, 0...nFFT / 2] = input.fft1D(direction: .forward, setup: plan)[0, 0...nFFT / 2]
    }

    if normalized {
        result /= sqrtf(Float(nFFT))
    }

    return result
}
