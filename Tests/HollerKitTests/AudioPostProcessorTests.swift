import Testing
@testable import HollerKit

@Suite("AudioPostProcessor")
struct AudioPostProcessorTests {

    @Test func fadeOutZerosLastSamples() {
        let samples = Array(repeating: Float(1.0), count: 24000) // 1s at 24kHz
        let faded = AudioPostProcessor.applyFadeOut(samples, fadeMs: 20, sampleRate: 24000)

        #expect(faded.count == samples.count)
        // Last sample should be ~0
        #expect(faded.last! < 0.01)
        // Sample well before fade region should be untouched
        #expect(faded[0] == 1.0)
        #expect(faded[23000] == 1.0)
    }

    @Test func fadeOutOnShortChunk() {
        let samples: [Float] = [1.0, 1.0, 1.0]
        let faded = AudioPostProcessor.applyFadeOut(samples, fadeMs: 20, sampleRate: 24000)
        // Chunk shorter than fade window — should still work, fade whole thing
        #expect(faded.count == 3)
        #expect(faded[2] < faded[0])
    }

    @Test func fadeOutOnEmpty() {
        let faded = AudioPostProcessor.applyFadeOut([], fadeMs: 20)
        #expect(faded.isEmpty)
    }

    @Test func fadeOutIsLinear() {
        let fadeSamples = 480 // 20ms at 24kHz
        let samples = Array(repeating: Float(1.0), count: fadeSamples)
        let faded = AudioPostProcessor.applyFadeOut(samples, fadeMs: 20, sampleRate: 24000)

        // First sample of fade region should be ~1.0, last should be ~0.0
        #expect(faded[0] > 0.99)
        #expect(faded[fadeSamples - 1] < 0.01)
        // Midpoint should be ~0.5
        let mid = faded[fadeSamples / 2]
        #expect(mid > 0.4 && mid < 0.6)
    }

    @Test func generateSilence() {
        let silence = AudioPostProcessor.generateSilence(durationMs: 100, sampleRate: 24000)
        #expect(silence.count == 2400)
        #expect(silence.allSatisfy { $0 == 0.0 })
    }
}
