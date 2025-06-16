// The Swift Programming Language
// https://docs.swift.org/swift-book

import CoreML
import Foundation

public func cqtFrequencies(
    nBins: Int, fMin: Float, binsPerOctave: Int = 12, tuning: Float = 0.0
)
    -> [Float]
{
    let correction = powf(2.0, tuning / Float(binsPerOctave))
    return (0..<nBins).map { i in
        correction * fMin * powf(2.0, Float(i) / Float(binsPerOctave))
    }
}
