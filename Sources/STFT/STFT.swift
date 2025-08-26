import Accelerate
import Foundation
import Numerics
import PFFFT
import Plinth

public func divceil(_ a: Int, _ b: Int) -> Int {
    let d = div(Int32(a), Int32(b))
    return Int(d.quot + (d.rem > 0 ? 1 : 0))
}

public func stft(
    signal: [Float],
    nFFT: Int,
    hopLength: Int? = nil,
    window: Windows.WindowType = .hann,
    center: Bool = true,
    padMode: PaddingType = .zeros,
    normalized: Bool = true
) -> ComplexMatrix<Float> {
    let hop = hopLength ?? nFFT / 4
    var inputSignal: Matrix<Float>
    if center {
        let n = signal.count
        let startK = divceil(nFFT / 2, hop)
        let tailK = (n + nFFT / 2 - nFFT) / hop + 1
        let padL = nFFT / 2
        let padR = (tailK <= startK || tailK * hop - nFFT / 2 * 2 + nFFT <= n) ? nFFT / 2 : 0
        let input = pad(x: signal, padSize: (padL, padR), type: padMode)
        inputSignal = .init(shape: .row(length: n + padL + padR), elements: input)
    } else {
        inputSignal = .init(shape: .row(length: signal.count), elements: signal)
    }

    let numSamples = inputSignal.count
    let numFrames = (numSamples - nFFT - 1) / hop + 1

    let w: [Float] = window == .ones ? [] : Windows.get(type: window, M: nFFT)

    var result: ComplexMatrix<Float> = .zeros(shape: .init(rows: numFrames, columns: nFFT / 2 + 1))

    DispatchQueue.concurrentPerform(iterations: numFrames) { i in
        guard let plan = try? PFFFT.FFT<Float>(n: nFFT) else {
            print("Failed to create FFT plan")
            return
        }
        let inputBuffer = plan.makeSignalBuffer()
        let outputBuffer = plan.makeSpectrumBuffer()

        inputBuffer.withUnsafeMutableBufferPointer { destBuffer in
            inputSignal.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!.advanced(by: i * hop),
                    destBuffer.baseAddress!,
                    1, vDSP_Length(nFFT),
                    1, 1)
            }
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

public func spectrogram(
    y: [Float],
    nFFT: Int = 2048,
    hopLength: Int? = 512,
    power: Float = 1,
    window: Windows.WindowType = .hann,
    center: Bool = true,
    padMode: PaddingType = .zeros
) -> Matrix<Float> {
    let spec = stft(
        signal: y, nFFT: nFFT,
        hopLength: hopLength, window: window,
        center: center, padMode: padMode
    )
    .absolute()
    return if power != 1 {
        spec ** power
    } else {
        spec
    }
}

public func istft(
    spec: ComplexMatrix<Float>,
    nFFT: Int? = nil,
    hopLength: Int? = nil,
    window: Windows.WindowType = .hann,
    center: Bool = true,
    length: Int? = nil,
    padMode: PaddingType = .zeros,
    normalized: Bool = true
) -> [Float] {
    // 1) Infer n_FFT and hop length
    let n_FFT = nFFT ?? 2 * (spec.shape.columns - 1)
    precondition(n_FFT % 2 == 0, "n_FFT must be even")
    let hop = hopLength ?? n_FFT / 4

    // 2) Determine how many frames we will actually use
    let nFrames: Int = spec.shape.rows

    // 3) Full overlap-add buffer length (before center-cropping)
    let fullOLALength: Int = n_FFT + hop * max(nFrames - 1, 0)

    // 4) Expected final length
    let expectedSignalLength: Int =
        if let L = length {
            L
        } else {
            center ? (fullOLALength - n_FFT) : fullOLALength
        }

    // 5) Prepare window, its square, and OLA accumulators
    let w: [Float] = window == .ones ? [] : Windows.get(type: window, M: n_FFT)
    let winSq = vDSP.square(w)

    var ola: [Float] = .init(repeating: 0, count: fullOLALength)
    var weight: [Float] = .init(repeating: 0, count: fullOLALength)

    // 6) FFT plan and buffers
    guard let plan = try? PFFFT.FFT<Float>(n: n_FFT) else {
        print("Failed to create FFT plan")
        return .init(repeating: 0, count: expectedSignalLength)
    }
    let timeBuffer = plan.makeSignalBuffer()  // length n_FFT
    let freqBuffer = plan.makeSpectrumBuffer()  // packed spectrum buffer (PFFFT real-FFT layout)

    // 7) Scaling to match stft(normalized) semantics and FFT normalization
    // - stft(normalized) divided by sqrt(n_FFT); to invert we multiply by sqrt(n_FFT)
    // - PFFFT inverse is typically unnormalized; we divide by n_FFT after inverse
    let scaleSpec: Float = normalized ? sqrtf(Float(n_FFT)) : 1.0
    let invFFTScale: Float = 1.0 / Float(n_FFT)

    // 8) Row-wise iFFT + OLA
    let rowStride = spec.shape.columns
    for i in 0..<nFrames {
        let start = i * hop

        // 8.1) Pack frequency bins into PFFFT's real-FFT layout
        spec.real.elements.withUnsafeBufferPointer { reBuf in
            spec.imaginary.elements.withUnsafeBufferPointer { imBuf in
                let reRow = reBuf.baseAddress!.advanced(by: i * rowStride)
                let imRow = imBuf.baseAddress!.advanced(by: i * rowStride)

                freqBuffer.withUnsafeMutableBufferPointer { fb in
                    let f = fb.baseAddress!
                    let realPtr = UnsafeMutableRawPointer(mutating: f)
                        .assumingMemoryBound(to: Float.self)
                    let imagPtr = realPtr.advanced(by: 1)

                    // DC and Nyquist
                    realPtr[0] = reRow[0]  // Re(0), Im(0)=0 implicitly
                    imagPtr[0] = reRow[n_FFT / 2]  // Re(N/2), Im(N/2)=0 implicitly

                    // Bins 1..N/2-1: interleaved Re, Im at (2k, 2k+1)
                    if n_FFT > 2 {
                        // Copy real parts: k=1..N/2-1 -> f[2k] = reRow[k]*scaleSpec
                        var k = 1
                        while k < (n_FFT / 2) {
                            f[k].real = reRow[k]
                            f[k].imaginary = imRow[k]
                            k += 1
                        }
                    }
                }
            }
        }

        // 8.2) Inverse FFT -> time domain
        plan.inverse(spectrum: freqBuffer, signal: timeBuffer)

        // 8.3) Normalize IFFT by N and apply synthesis window, then OLA
        timeBuffer.withUnsafeMutableBufferPointer { tb in
            let tptr = tb.baseAddress!

            // Apply window and overlap-add with weight accumulation
            // y[start + j] += t[j] * w[j]
            // weight[start + j] += w[j]^2
            if window == .ones {
                // Window == ones: faster path
                var j = 0
                while j < n_FFT {
                    ola[start + j] += tptr[j]
                    weight[start + j] += 1
                    j += 1
                }
            } else {
                var j = 0
                while j < n_FFT {
                    let v = tptr[j] * w[j]
                    ola[start + j] += v
                    weight[start + j] += winSq[j]
                    j += 1
                }
            }
        }
    }

    // 9) Divide by window square sum with epsilon guard
    let eps: Float = 1e-10
    var outFull = ola  // reuse buffer
    let count = fullOLALength
    outFull.withUnsafeMutableBufferPointer { yb in
        weight.withUnsafeMutableBufferPointer { wb in
            let y = yb.baseAddress!
            let w = wb.baseAddress!
            var i = 0
            while i < count {
                let denom = w[i]
                if denom > eps {
                    y[i] /= denom
                } else {
                    // Where denominator is (near) zero, leave as is (0) per common practice
                    y[i] = 0
                }
                i += 1
            }
        }
    }

    // 10) Center-cropping and final length adjustment
    var output: [Float]
    if center {
        let cropStart = n_FFT / 2
        let cropEnd = min(cropStart + expectedSignalLength, outFull.count)
        if cropStart < cropEnd {
            output = Array(outFull[cropStart..<cropEnd])
        } else {
            output = []
        }
    } else {
        // No center: start from the beginning
        output = outFull
        if output.count > expectedSignalLength {
            output.removeLast(output.count - expectedSignalLength)
        }
    }

    // If we still need to pad to 'length'
    if output.count < expectedSignalLength {
        output.append(contentsOf: repeatElement(0, count: expectedSignalLength - output.count))
    }

    vDSP.multiply(scaleSpec * invFFTScale, output, result: &output)

    return output
}
