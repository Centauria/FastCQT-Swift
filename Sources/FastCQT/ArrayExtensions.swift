import Accelerate

extension Array where Element == Float {
    func median() -> Float? {
        guard count > 0 else { return nil }

        let sortedArray = self.sorted()
        return if count % 2 != 0 {
            sortedArray[count / 2]
        } else {
            (sortedArray[count / 2] + sortedArray[count / 2 - 1]) / 2.0
        }
    }

    func histogram(bins: [Float]) -> [Int] {
        precondition(bins.count > 1)
        precondition(bins.isStrictlyAscending)
        var count = [Int](repeating: 0, count: bins.count - 1)
        for x in self {
            if let index = bins.firstIndex(where: { x <= $0 }), index > 0 {
                count[index - 1] += 1
            }
        }
        return count
    }
}

extension Array where Element: Comparable {
    var isStrictlyAscending: Bool {
        return isEmpty || count == 1 || zip(self, dropFirst()).allSatisfy { $0 < $1 }
    }

    var isAscending: Bool {
        return isEmpty || count == 1 || zip(self, dropFirst()).allSatisfy { $0 <= $1 }
    }
}
