public enum AudioPostProcessor {

    /// Apply a linear fade-out to the end of a chunk.
    /// Port of server.py `_apply_trailing_fadeout()` L207-214.
    public static func applyFadeOut(
        _ samples: [Float],
        fadeMs: Float = 20,
        sampleRate: Int = 24000
    ) -> [Float] {
        var fadeSamples = Int(Float(sampleRate) * fadeMs / 1000)
        if samples.count < fadeSamples {
            fadeSamples = samples.count
        }
        guard fadeSamples > 0 else { return samples }

        var result = samples
        let fadeStart = result.count - fadeSamples
        for i in 0..<fadeSamples {
            let gain = 1.0 - Float(i) / Float(fadeSamples)
            result[fadeStart + i] *= gain
        }
        return result
    }

    /// Generate silence of a given duration.
    public static func generateSilence(
        durationMs: Int,
        sampleRate: Int = 24000
    ) -> [Float] {
        Array(repeating: 0.0, count: sampleRate * durationMs / 1000)
    }
}
