import XCTest

@testable import FastCQT

final class ArrayExtensionsTests: XCTestCase {
    func testMedian() throws {
        let x: [Float] = [1.3, 2.1, 3, 2, 9]
        let y = x.median()!
        assert(y == 2.1)
        let x1: [Float] = [1.3, 2.1, 3, 2, 9, 2.5]
        let y1 = x1.median()!
        assert(y1 == 2.3)
        let x2: [Float] = []
        let y2 = x2.median()
        assert(y2 == nil)
    }

    func testHistogram() throws {
        let x: [Float] = [
            0.77505197, 0.9599713, 0.22603799, 0.75261799, 0.21095125,
            0.03115751, 0.27621601, 0.2994032, 0.41304219, 0.92051206,
            0.11340166, 0.27630736, 0.82167019, 0.04333095, 0.03576075,
            0.06041459, 0.14715272, 0.85320324, 0.33045069, 0.89577716,
        ]
        let g: [Float] = [
            0.11425171, 0.15099409, 0.15254688, 0.25812283, 0.27361185,
            0.39582451, 0.6680375, 0.94305063, 0.9765229, 0.99080513,
        ]
        let y = x.histogram(bins: g)
        assert(y == [1, 0, 2, 0, 4, 1, 6, 1, 0])
    }
}
