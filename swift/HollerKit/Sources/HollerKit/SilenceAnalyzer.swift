import Foundation

public enum SilenceAnalyzer {

    public static let defaultSampleRate: Int = 24000

    /// Check if a chunk contains speech using 2-of-3 temporal RMS confirmation.
    /// Port of server.py `_has_speech()` L173-183.
    public static func hasSpeech(
        _ samples: [Float],
        threshold: Float = 0.007,
        sampleRate: Int = defaultSampleRate
    ) -> Bool {
        let windowSize = Int(Double(sampleRate) * 0.01) // 10ms
        guard windowSize > 0, samples.count >= windowSize else { return false }

        var recent = [false, false, false]
        var windowIndex = 0
        var j = 0
        while j + windowSize <= samples.count {
            let rms = rmsOfWindow(samples, start: j, length: windowSize)
            recent[windowIndex % 3] = rms >= threshold
            if recent.filter({ $0 }).count >= 2 {
                return true
            }
            windowIndex += 1
            j += windowSize
        }
        return false
    }

    /// Find speech onset sample index with pre-roll.
    /// Scans 10ms windows. Speech confirmed when 2 of 3 consecutive windows
    /// exceed the RMS threshold. Returns sample index with pre-roll applied,
    /// or nil if no speech found.
    /// Port of server.py `_find_speech_onset()` L186-204.
    public static func findSpeechOnset(
        _ samples: [Float],
        threshold: Float = 0.007,
        preRollMs: Float = 150,
        sampleRate: Int = defaultSampleRate
    ) -> Int? {
        let windowSize = Int(Double(sampleRate) * 0.01)
        let preRollSamples = Int(Double(sampleRate) * Double(preRollMs) / 1000.0)
        guard windowSize > 0, samples.count >= windowSize else { return nil }

        var recent = [false, false, false]
        var windowIndex = 0
        var j = 0
        while j + windowSize <= samples.count {
            let rms = rmsOfWindow(samples, start: j, length: windowSize)
            recent[windowIndex % 3] = rms >= threshold
            if recent.filter({ $0 }).count >= 2 {
                let firstWindow = max(0, j - 2 * windowSize)
                return max(0, firstWindow - preRollSamples)
            }
            windowIndex += 1
            j += windowSize
        }
        return nil
    }

    private static func rmsOfWindow(_ samples: [Float], start: Int, length: Int) -> Float {
        var sumSquares: Float = 0
        let end = min(start + length, samples.count)
        for i in start..<end {
            sumSquares += samples[i] * samples[i]
        }
        let mean = sumSquares / Float(end - start)
        return mean.squareRoot()
    }
}
