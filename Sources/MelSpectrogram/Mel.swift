import Accelerate
import Plinth

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
        return 2595.0 * log10(1.0 + freq / 700)
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

func melscaleBanks(
    nFreqs: Int,
    fMin: Float,
    fMax: Float,
    nMels: Int,
    sampleRate: Float,
    slaney: Bool
) -> Matrix<Float> {
    let allFreqs = vDSP.ramp(in: 0...floor(sampleRate / 2), count: nFreqs)
    return .empty
}
