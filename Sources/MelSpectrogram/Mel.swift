import Accelerate
import Plinth
import STFT

func _hzToMel(freq: Float, slaney: Bool) -> Float {
    if slaney {
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3
        let mels: Float
        let minLogHz: Float = 1000
        if freq >= minLogHz {
            let minLogMel = (minLogHz - fMin) / fSp
            let logStep: Float = log(6.4) / 27
            mels = minLogMel + log(freq / minLogHz) / logStep
        } else {
            mels = (freq - fMin) / fSp
        }
        return mels
    } else {
        return 2595.0 / log(10) * log1p(freq / 700)
    }
}

func _melToHz(mels: [Float], slaney: Bool) -> [Float] {
    if slaney {
        let fMin: Float = 0.0
        let fSp: Float = 200.0 / 3
        var freqs = vDSP.add(multiplication: (mels, fSp), fMin)

        let minLogHz: Float = 1000
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep: Float = log(6.4) / 27

        for i in 0..<mels.count {
            if mels[i] >= minLogMel {
                freqs[i] = minLogHz * exp(logStep * (mels[i] - minLogMel))
            }
        }
        return freqs
    } else {
        let slope: Float = log(10.0) / 2595.0
        var freqs = vDSP.multiply(slope, mels)
        vForce.expm1(freqs, result: &freqs)
        vDSP.multiply(700, freqs, result: &freqs)
        return freqs
    }
}

func _createTriangularFilterbank(allFreqs: [Float], fPts: [Float]) -> Matrix<Float> {
    let nFreqs = allFreqs.count
    let nfDiff = fPts.count - 1
    let nFilters = nfDiff - 1
    let fDiff: [Float] = .init(unsafeUninitializedCapacity: nfDiff) { buffer, initializedCount in
        let fd = buffer.baseAddress!
        fPts.withUnsafeBufferPointer {
            let fp = $0.baseAddress!
            vDSP_vsub(fp, 1, fp.advanced(by: 1), 1, fd, 1, vDSP_Length(nfDiff))
        }
        initializedCount = nfDiff
    }
    let slopes: Matrix<Float> = .init(shape: .init(rows: nFreqs, columns: fPts.count)) { i, j in
        fPts[j] - allFreqs[i]
    }
    let downSlopes: Matrix<Float> = .init(shape: .init(rows: nFreqs, columns: nFilters)) { i, j in
        -slopes[i, j] / fDiff[j]
    }
    let upSlopes: Matrix<Float> = .init(shape: .init(rows: nFreqs, columns: nFilters)) { i, j in
        slopes[i, j + 2] / fDiff[j + 1]
    }
    let fb = downSlopes.minimum(upSlopes).threshold(to: 0, with: .clampToThreshold)
    return fb
}

public func melscaleFreqBanks(
    nFreqs: Int,
    fMin: Float,
    fMax: Float,
    nMels: Int,
    sampleRate: Float,
    norm: Bool = false,
    slaney: Bool = true
) -> Matrix<Float> {
    let allFreqs = vDSP.ramp(in: 0...floor(sampleRate / 2), count: nFreqs)
    let mMin = _hzToMel(freq: fMin, slaney: slaney)
    let mMax = _hzToMel(freq: fMax, slaney: slaney)
    let mPts = vDSP.ramp(in: mMin...mMax, count: nMels + 2)
    let fPts = _melToHz(mels: mPts, slaney: slaney)
    var fb = _createTriangularFilterbank(allFreqs: allFreqs, fPts: fPts)
    if norm {
        let enorm = vDSP.divide(2.0, vDSP.subtract(fPts[2..<nMels + 2], fPts[0..<nMels]))
        for i in 0..<nMels {
            fb[0..<nFreqs, i] *= enorm[i]
        }
    }
    return fb
}

public func melSpect(
    y: [Float],
    sampleRate: Float = 16000,
    nFFT: Int = 400,
    winLength: Int? = nil,
    hopLength: Int? = nil,
    fMin: Float = 0,
    fMax: Float? = nil,
    nMels: Int = 128,
    window: Windows.WindowType = .hann,
    power: Float = 2,
    center: Bool = true,
    norm: Bool = false,
    slaney: Bool = true
) -> Matrix<Float> {
    let spec = spectrogram(
        y: y, nFFT: nFFT, hopLength: hopLength,
        power: power, window: window, center: center,
        padMode: .reflect)
    let mel = melscaleFreqBanks(
        nFreqs: nFFT / 2 + 1,
        fMin: fMin, fMax: fMax ?? floor(sampleRate / 2),
        nMels: nMels, sampleRate: sampleRate,
        norm: norm, slaney: slaney)
    let melSpec = spec <*> mel
    return melSpec
}

public func logMelSpect(
    y: [Float],
    logMultiplier: Float = 1000,
    sampleRate: Float = 16000,
    nFFT: Int = 400,
    winLength: Int? = nil,
    hopLength: Int? = nil,
    fMin: Float = 0,
    fMax: Float? = nil,
    nMels: Int = 128,
    power: Float = 2,
    norm: Bool = false,
    slaney: Bool = true
) -> Matrix<Float> {
    var melSpec = melSpect(
        y: y, sampleRate: sampleRate, nFFT: nFFT, winLength: winLength, hopLength: hopLength,
        fMin: fMin, fMax: fMax, nMels: nMels, window: .hann, power: power,
        center: true, norm: norm, slaney: slaney)
    melSpec *= logMultiplier
    return melSpec.log1p()
}
