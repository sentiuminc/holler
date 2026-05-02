import Testing
@testable import HollerKit

@Suite("AudioPostProcessor")
struct AudioPostProcessorTests {

    @Test func generateSilence() {
        let silence = AudioPostProcessor.generateSilence(durationMs: 100, sampleRate: 24000)
        #expect(silence.count == 2400)
        #expect(silence.allSatisfy { $0 == 0.0 })
    }
}
