// The Swift Programming Language
// https://docs.swift.org/swift-book

import CoreML
import Foundation

func cqtFrequencies(
    nBins: Int, fMin: Float32, binsPerOctave: Int = 12, tuning: Float32 = 0.0
)
    -> [Float32]
{
    let correction = powf(2.0, tuning / Float32(binsPerOctave))
    return (0..<nBins).map { i in
        correction * fMin * powf(2.0, Float32(i) / Float32(binsPerOctave))
    }
}