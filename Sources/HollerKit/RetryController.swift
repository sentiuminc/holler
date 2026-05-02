public struct RetryController: Sendable {
    public let maxRetries: Int
    public let temperatureStep: Float
    public let minAudioSecondsPerWord: Float

    public init(config: HollerConfiguration) {
        self.maxRetries = config.maxRetries
        self.temperatureStep = config.retryTemperatureStep
        self.minAudioSecondsPerWord = config.minAudioSecondsPerWord
    }

    public enum Decision: Equatable, Sendable {
        case accept
        case retry(temperature: Float)
        case giveUp
    }

    /// Evaluate whether a generation attempt should be accepted, retried, or abandoned.
    /// Port of server.py `generate_audio()` L424-484 retry logic.
    public func evaluate(
        attempt: Int,
        baseTemperature: Float,
        totalSamples: Int,
        sampleRate: Int,
        wordCount: Int
    ) -> Decision {
        let minSamples = Int(Float(wordCount) * minAudioSecondsPerWord * Float(sampleRate))

        if totalSamples >= minSamples {
            return .accept
        }

        guard attempt < maxRetries else {
            return .giveUp
        }

        let nextTemp = min(baseTemperature + Float(attempt + 1) * temperatureStep, 1.0)
        return .retry(temperature: nextTemp)
    }
}
