import XCTest
@testable import CaptionWindow

final class EscapeCloseGateTests: XCTestCase {
    func testSecondEscapeWithinIntervalRequestsClose() {
        var gate = EscapeCloseGate(confirmInterval: 1.0)

        XCTAssertFalse(gate.shouldClose(at: 10.0))
        XCTAssertTrue(gate.shouldClose(at: 10.5))
    }

    func testSecondEscapeAfterIntervalDoesNotRequestClose() {
        var gate = EscapeCloseGate(confirmInterval: 1.0)

        XCTAssertFalse(gate.shouldClose(at: 10.0))
        XCTAssertFalse(gate.shouldClose(at: 11.5))
        XCTAssertTrue(gate.shouldClose(at: 12.0))
    }
}
