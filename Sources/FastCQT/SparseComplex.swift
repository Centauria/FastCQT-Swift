import Accelerate
import Numerics

public struct SparseMatrix_ComplexFloat {
    var _real: SparseMatrix_Float
    var _imag: [Float]

    public var real: SparseMatrix_Float {
        _real
    }

    public var imag: SparseMatrix_Float {
        _imag.withUnsafeBufferPointer { buf in
            let mutBuf = UnsafeMutablePointer<Float>(mutating: buf.baseAddress)
            return .init(structure: real.structure, data: mutBuf!)
        }
    }
}

extension SparseMatrix_Float: CustomStringConvertible {
    public var description: String {
        let rows = structure.rowCount
        let cols = structure.columnCount
        let nnz = structure.columnStarts[Int(cols)]

        return "SparseMatrix_Float(\(rows)×\(cols), \(nnz) non-zeros)"
    }
}

extension SparseMatrix_Float {
    /// 安全地获取元素，返回Optional
    public func element(at row: Int, column: Int) -> Float? {
        guard row >= 0 && row < structure.rowCount,
            column >= 0 && column < structure.columnCount
        else {
            return nil  // 越界返回nil
        }

        let colStart = Int(structure.columnStarts[column])
        let colEnd = Int(structure.columnStarts[column + 1])

        for i in colStart..<colEnd {
            if Int(structure.rowIndices[i]) == row {
                return data[i]
            }
        }

        return 0.0  // 稀疏矩阵中的隐式零值
    }

    /// 获取整个列的非零元素
    public func column(_ col: Int) -> [(row: Int, value: Float)] {
        guard col >= 0 && col < structure.columnCount else {
            return []
        }

        let colStart = Int(structure.columnStarts[col])
        let colEnd = Int(structure.columnStarts[col + 1])

        var result: [(row: Int, value: Float)] = []
        for i in colStart..<colEnd {
            let row = Int(structure.rowIndices[i])
            let value = data[i]
            result.append((row: row, value: value))
        }

        return result
    }

    /// 获取所有非零元素
    public func nonZeroElements() -> [(row: Int, column: Int, value: Float)] {
        var result: [(row: Int, column: Int, value: Float)] = []

        for col in 0..<Int(structure.columnCount) {
            let colStart = Int(structure.columnStarts[col])
            let colEnd = Int(structure.columnStarts[col + 1])

            for i in colStart..<colEnd {
                let row = Int(structure.rowIndices[i])
                let value = data[i]
                result.append((row: row, column: col, value: value))
            }
        }

        return result
    }
}

extension SparseMatrix_ComplexFloat: CustomStringConvertible {
    public var description: String {
        let rows = real.structure.rowCount
        let cols = real.structure.columnCount
        let nnz = real.structure.columnStarts[Int(cols)]

        return "SparseMatrix_ComplexFloat(\(rows)×\(cols), \(nnz) non-zeros)"
    }
}

extension SparseMatrix_ComplexFloat {
    /// 安全地获取元素，返回Optional
    public func element(at row: Int, column: Int) -> Complex<Float>? {
        let structure = real.structure
        guard row >= 0 && row < structure.rowCount,
            column >= 0 && column < structure.columnCount
        else {
            return nil  // 越界返回nil
        }

        let colStart = Int(structure.columnStarts[column])
        let colEnd = Int(structure.columnStarts[column + 1])

        for i in colStart..<colEnd {
            if Int(structure.rowIndices[i]) == row {
                return .init(real.data[i], _imag[i])
            }
        }

        return .init(0, 0)
    }

    /// 获取整个列的非零元素
    public func column(_ col: Int) -> [(row: Int, value: Complex<Float>)] {
        let structure = real.structure
        guard col >= 0 && col < structure.columnCount else {
            return []
        }

        let colStart = Int(structure.columnStarts[col])
        let colEnd = Int(structure.columnStarts[col + 1])

        var result: [(row: Int, value: Complex<Float>)] = []
        for i in colStart..<colEnd {
            let row = Int(structure.rowIndices[i])
            let value: Complex<Float> = .init(real.data[i], _imag[i])
            result.append((row: row, value: value))
        }

        return result
    }

    /// 获取所有非零元素
    public func nonZeroElements() -> [(row: Int, column: Int, value: Complex<Float>)] {
        let structure = real.structure
        var result: [(row: Int, column: Int, value: Complex<Float>)] = []

        for col in 0..<Int(structure.columnCount) {
            let colStart = Int(structure.columnStarts[col])
            let colEnd = Int(structure.columnStarts[col + 1])

            for i in colStart..<colEnd {
                let row = Int(structure.rowIndices[i])
                let value: Complex<Float> = .init(real.data[i], _imag[i])
                result.append((row: row, column: col, value: value))
            }
        }

        return result
    }
}

extension SparseMatrix_ComplexFloat {
    public func absolute() -> SparseMatrix_Float {
        let structure = real.structure
        let r = Array(UnsafeBufferPointer(start: real.data, count: _imag.count))
        var a = vDSP.hypot(r, _imag)
        return a.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!
            return .init(structure: structure, data: ptr)
        }
    }
}
