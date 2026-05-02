// No production callers currently — kept as utility infrastructure.
public enum AudioPostProcessor {

    /// Generate silence of a given duration.
    public static func generateSilence(
        durationMs: Int,
        sampleRate: Int = 24000
    ) -> [Float] {
        Array(repeating: 0.0, count: sampleRate * durationMs / 1000)
    }
}
