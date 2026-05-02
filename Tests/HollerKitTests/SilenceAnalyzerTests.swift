import Foundation
import Testing
@testable import HollerKit

@Suite("SilenceAnalyzer")
struct SilenceAnalyzerTests {

    let sampleRate = 24000

    func makeSilence(durationMs: Int) -> [Float] {
        Array(repeating: 0.0, count: sampleRate * durationMs / 1000)
    }

    func makeTone(durationMs: Int, amplitude: Float = 0.5, frequencyHz: Float = 440) -> [Float] {
        let count = sampleRate * durationMs / 1000
        return (0..<count).map { i in
            let phase = 2.0 * Float.pi * frequencyHz * Float(i) / Float(sampleRate)
            return amplitude * Foundation.sinf(phase)
        }
    }

    // MARK: - hasSpeech

    @Test func silenceIsNotSpeech() {
        let silence = makeSilence(durationMs: 100)
        #expect(!SilenceAnalyzer.hasSpeech(silence, sampleRate: sampleRate))
    }

    @Test func loudToneIsSpeech() {
        let tone = makeTone(durationMs: 100, amplitude: 0.5)
        #expect(SilenceAnalyzer.hasSpeech(tone, sampleRate: sampleRate))
    }

    @Test func quietToneIsNotSpeech() {
        let quiet = makeTone(durationMs: 100, amplitude: 0.001)
        #expect(!SilenceAnalyzer.hasSpeech(quiet, sampleRate: sampleRate))
    }

    @Test func emptyArrayIsNotSpeech() {
        #expect(!SilenceAnalyzer.hasSpeech([]))
    }

    @Test func twoOfThreeWindowConfirmation() {
        // One loud window sandwiched by silence shouldn't trigger (only 1 of 3)
        // But two loud windows should trigger
        var samples = makeSilence(durationMs: 10)
        samples += makeTone(durationMs: 10, amplitude: 0.5)
        samples += makeSilence(durationMs: 10)
        // 1 of 3 windows is loud — should NOT detect speech
        #expect(!SilenceAnalyzer.hasSpeech(samples, sampleRate: sampleRate))

        // Two consecutive loud windows — should detect
        var twoLoud = makeTone(durationMs: 20, amplitude: 0.5)
        twoLoud += makeSilence(durationMs: 10)
        #expect(SilenceAnalyzer.hasSpeech(twoLoud, sampleRate: sampleRate))
    }

    // MARK: - findSpeechOnset

    @Test func onsetInPureSilenceReturnsNil() {
        let silence = makeSilence(durationMs: 200)
        #expect(SilenceAnalyzer.findSpeechOnset(silence, sampleRate: sampleRate) == nil)
    }

    @Test func onsetAtStartWithPreRoll() {
        let tone = makeTone(durationMs: 100, amplitude: 0.5)
        let onset = SilenceAnalyzer.findSpeechOnset(tone, sampleRate: sampleRate)
        #expect(onset != nil)
        #expect(onset == 0) // Can't go before start
    }

    @Test func onsetAfterSilenceIncludesPreRoll() {
        // 300ms silence then speech — onset should be ~150ms (300ms - 150ms preroll)
        var samples = makeSilence(durationMs: 300)
        samples += makeTone(durationMs: 100, amplitude: 0.5)
        let onset = SilenceAnalyzer.findSpeechOnset(samples, preRollMs: 150, sampleRate: sampleRate)
        #expect(onset != nil)
        // Onset should be well before the speech starts (300ms mark = sample 7200)
        // but after sample 0 because pre-roll from 300ms position
        if let onset {
            #expect(onset > 0)
            #expect(onset < 7200)
        }
    }

    @Test func onsetOnEmptyReturnsNil() {
        #expect(SilenceAnalyzer.findSpeechOnset([], sampleRate: sampleRate) == nil)
    }
}
