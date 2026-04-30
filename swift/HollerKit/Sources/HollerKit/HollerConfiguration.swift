public struct HollerConfiguration: Sendable {
    public var temperature: Float = 0.6
    public var topK: Int = 50
    public var codebooks: Int = 12
    public var maxTokens: Int = 500

    public var streamingChunkTokens: Int = 3

    // Silence handling
    public var silentAbortTokens: Int = 16
    public var speechOnsetThresholdRMS: Float = 0.007
    public var speechOnsetPreRollMs: Float = 150
    public var fadeOutMs: Float = 20

    // Retry
    public var maxRetries: Int = 3
    public var retryTemperatureStep: Float = 0.1
    public var minAudioSecondsPerWord: Float = 0.08

    // Carryover (Phase 2)
    public var carryoverPauseMinMs: Int = 150
    public var carryoverPauseMaxMs: Int = 250

    public init() {}
}
