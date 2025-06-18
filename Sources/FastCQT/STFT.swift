import Accelerate
import Foundation
import Numerics
import PFFFT
import Plinth

public func stft(
    signal: [Float],
    nFFT: Int,
    hopLength: Int,
    window: Windows.WindowType = .hann,
    center: Bool = true,
    normalized: Bool = true
) -> ComplexMatrix<Float>? {
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

    let w: [Float] = window == .ones ? [] : Windows.get(type: window, M: nFFT)

    var result: ComplexMatrix<Float> = .zeros(shape: .init(rows: numFrames, columns: nFFT / 2 + 1))

    DispatchQueue.concurrentPerform(iterations: numFrames) { i in
        guard let plan = try? PFFFT.FFT<Float>(n: nFFT) else {
            print("Failed to create FFT plan")
            return
        }
        let inputBuffer = plan.makeSignalBuffer()
        let outputBuffer = plan.makeSpectrumBuffer()

        let length = lastFrameComplete || i == numFrames ? numSamples - i * hopLength : nFFT
        inputBuffer.withUnsafeMutableBufferPointer { destBuffer in
            inputSignal.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!.advanced(by: i * hopLength),
                    destBuffer.baseAddress!,
                    1, vDSP_Length(length),
                    1, 1)
            }
            vDSP_vclr(destBuffer.baseAddress!, 1, vDSP_Length(nFFT - length))
        }
        if window != .ones {
            inputBuffer.withUnsafeMutableBufferPointer { destBuffer in
                let y = destBuffer.baseAddress!
                w.withUnsafeBufferPointer { winBuffer in
                    vDSP_vmul(
                        y, 1,
                        winBuffer.baseAddress!, 1,
                        y, 1,
                        vDSP_Length(nFFT))
                }
            }
        }
        plan.forward(signal: inputBuffer, spectrum: outputBuffer)
        outputBuffer.withUnsafeBufferPointer { srcBuffer in
            let bufferPtr = srcBuffer.baseAddress!
            let realPtr = UnsafeMutableRawPointer(mutating: bufferPtr)
                .assumingMemoryBound(to: Float.self)
            let imagPtr = realPtr.advanced(by: 1)
            result.real.elements.withUnsafeMutableBufferPointer { destBuffer in
                let currentRowPtr = destBuffer.baseAddress!.advanced(by: (nFFT / 2 + 1) * i)
                vDSP_mmov(realPtr, currentRowPtr, 1, vDSP_Length(nFFT / 2), 2, 1)
                // currentRowPtr.advanced(by: nFFT / 2)
                //     .update(from: realPtr.advanced(by: 1), count: 1)
                currentRowPtr[nFFT / 2] = realPtr[1]
            }
            result.imaginary.elements.withUnsafeMutableBufferPointer { destBuffer in
                let currentRowPtr = destBuffer.baseAddress!.advanced(by: (nFFT / 2 + 1) * i)
                vDSP_mmov(
                    imagPtr.advanced(by: 2),
                    currentRowPtr.advanced(by: 1),
                    1, vDSP_Length(nFFT / 2 - 1),
                    2, 1)
            }
        }
    }

    if normalized {
        result /= sqrtf(Float(nFFT))
    }

    return result
}
