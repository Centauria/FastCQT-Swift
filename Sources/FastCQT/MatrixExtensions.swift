import Accelerate
import Plinth

extension Matrix {
    @inlinable public mutating func fmapInplace(_ function: (inout [Scalar]) -> Void) {
        function(&elements)
    }
}

extension Matrix where Scalar == Float {
    public func maximum(axis: Int) -> Matrix {
        precondition((0...1).contains(axis))
        let m = shape.rows
        let n = shape.columns
        return if axis == 0 {
            .init(
                shape: .row(length: n),
                elements: [Scalar](unsafeUninitializedCapacity: n) {
                    yptr, count in
                    let y = yptr.baseAddress!
                    elements.withUnsafeBufferPointer {
                        let x = $0.baseAddress!
                        for i in 0..<n {
                            vDSP_maxv(x.advanced(by: i), n, y.advanced(by: i), vDSP_Length(m))
                        }
                    }
                    count = n
                }
            )
        } else {
            .init(
                shape: .column(length: m),
                elements: [Scalar](unsafeUninitializedCapacity: m) {
                    yptr, count in
                    let y = yptr.baseAddress!
                    elements.withUnsafeBufferPointer {
                        let x = $0.baseAddress!
                        for i in 0..<m {
                            vDSP_maxv(x.advanced(by: i * n), 1, y.advanced(by: i), vDSP_Length(n))
                        }
                    }
                    count = m
                }
            )
        }
    }

    public func threshold(to lowerBound: Matrix, with rule: vDSP.ThresholdRule<Float>) -> Matrix {
        let bRow = lowerBound.shape.rows == 1 && lowerBound.shape.columns == shape.columns
        let bCol = lowerBound.shape.rows == shape.rows && lowerBound.shape.columns == 1
        precondition(bRow || bCol)
        let m = shape.rows
        let n = shape.columns
        let f =
            switch rule {
            case .clampToThreshold:
                vDSP_vthr
            case .signedConstant(let replacement):
                {
                    var C = replacement
                    vDSP_vthrsc($0, $1, $2, &C, $3, $4, $5)
                }
            case .zeroFill:
                vDSP_vthres
            @unknown default:
                vDSP_vthr
            }
        return .init(
            shape: shape,
            elements: [Scalar](unsafeUninitializedCapacity: count) {
                yptr, initializedNum in
                let y = yptr.baseAddress!
                elements.withUnsafeBufferPointer {
                    let x = $0.baseAddress!
                    lowerBound.elements.withUnsafeBufferPointer {
                        let th = $0.baseAddress!
                        if bRow {
                            for i in 0..<n {
                                f(
                                    x.advanced(by: i), n,
                                    th.advanced(by: i),
                                    y.advanced(by: i), n,
                                    vDSP_Length(m))
                            }
                        } else {
                            for i in 0..<m {
                                f(
                                    x.advanced(by: i * n), 1,
                                    th.advanced(by: i),
                                    y.advanced(by: i * n), 1,
                                    vDSP_Length(n))
                            }
                        }
                    }
                }
                initializedNum = count
            }
        )
    }

    public mutating func thresholdInplace(
        to lowerBound: Matrix, with rule: vDSP.ThresholdRule<Float>
    ) {
        let bRow = lowerBound.shape.rows == 1 && lowerBound.shape.columns == shape.columns
        let bCol = lowerBound.shape.rows == shape.rows && lowerBound.shape.columns == 1
        precondition(bRow || bCol)
        let m = shape.rows
        let n = shape.columns
        let f =
            switch rule {
            case .clampToThreshold:
                vDSP_vthr
            case .signedConstant(let replacement):
                {
                    var C = replacement
                    vDSP_vthrsc($0, $1, $2, &C, $3, $4, $5)
                }
            case .zeroFill:
                vDSP_vthres
            @unknown default:
                vDSP_vthr
            }
        elements.withUnsafeMutableBufferPointer {
            let x = $0.baseAddress!
            lowerBound.elements.withUnsafeBufferPointer {
                let th = $0.baseAddress!
                if bRow {
                    for i in 0..<n {
                        f(
                            x.advanced(by: i), n,
                            th.advanced(by: i),
                            x.advanced(by: i), n,
                            vDSP_Length(m))
                    }
                } else {
                    for i in 0..<m {
                        f(
                            x.advanced(by: i * n), 1,
                            th.advanced(by: i),
                            x.advanced(by: i * n), 1,
                            vDSP_Length(n))
                    }
                }
            }
        }
    }

    public mutating func thresholdInplace(
        to lowerBound: Scalar, with rule: vDSP.ThresholdRule<Float>
    ) {
        fmapInplace { vDSP.threshold($0, to: lowerBound, with: rule, result: &$0) }
    }
}
