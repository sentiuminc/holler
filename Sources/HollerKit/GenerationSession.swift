/// Processes raw audio through the silence pipeline.
/// Port of server.py silence handling from `_run_generation()`.
///
/// Applied to complete audio (batch generation):
/// 1. Onset trim — find speech start with 150ms pre-roll
/// 2. Trailing silence trim — find speech end, trim silence
struct GenerationSession {

    struct Result: Sendable {
        let samples: [Float]
        let aborted: Bool
    }

    static func process(
        rawSamples: [Float],
        config: HollerConfiguration,
        sampleRate: Int = 24000
    ) -> Result {
        guard !rawSamples.isEmpty else {
            return Result(samples: [], aborted: true)
        }

        // Onset trim: find where speech starts
        guard let onset = SilenceAnalyzer.findSpeechOnset(
            rawSamples,
            threshold: config.speechOnsetThresholdRMS,
            preRollMs: config.speechOnsetPreRollMs,
            sampleRate: sampleRate
        ) else {
            return Result(samples: [], aborted: true)
        }

        var trimmed = Array(rawSamples[onset...])

        // Trailing silence trim: scan backwards for last speech
        let windowSize = Int(Double(sampleRate) * 0.01)
        var lastSpeechEnd = trimmed.count
        if windowSize > 0 {
            let preRollSamples = Int(Double(sampleRate) * 0.05) // 50ms trailing buffer
            var j = trimmed.count
            while j >= windowSize {
                j -= windowSize
                let windowSlice = Array(trimmed[j..<min(j + windowSize, trimmed.count)])
                if SilenceAnalyzer.hasSpeech(windowSlice, threshold: config.speechOnsetThresholdRMS, sampleRate: sampleRate) {
                    lastSpeechEnd = min(trimmed.count, j + windowSize + preRollSamples)
                    break
                }
            }
        }

        if lastSpeechEnd < trimmed.count {
            trimmed = Array(trimmed[..<lastSpeechEnd])
        }

        return Result(samples: trimmed, aborted: false)
    }
}
