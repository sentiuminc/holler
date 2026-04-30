import Foundation

public final class HollerModel: @unchecked Sendable {
    private let actor: InferenceActor
    public var configuration: HollerConfiguration

    private init(actor: InferenceActor, configuration: HollerConfiguration) {
        self.actor = actor
        self.configuration = configuration
    }

    /// Load a Holler model from a HuggingFace repo.
    /// Downloads on first use, cached for subsequent loads.
    public static func load(
        repo: String = "sentium/holler-0.6b-6bit",
        configuration: HollerConfiguration = HollerConfiguration()
    ) async throws -> HollerModel {
        let actor = InferenceActor()
        try await actor.loadModel(repo: repo)
        return HollerModel(actor: actor, configuration: configuration)
    }

    public var voices: [String] {
        get async { await actor.voices }
    }

    public var isLoaded: Bool {
        get async { await actor.isLoaded }
    }

    public var sampleRate: Int {
        get async { await actor.sampleRate }
    }

    /// Stream production-quality audio (silence-trimmed, faded, retried).
    /// Yields a single chunk containing the complete processed audio.
    public func stream(
        _ text: String,
        voice: String
    ) -> AsyncThrowingStream<HollerAudioChunk, Error> {
        let config = self.configuration
        let actor = self.actor
        let wordCount = text.split(separator: " ").count

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let voices = await actor.voices
                    if !voices.isEmpty, !voices.contains(voice) {
                        throw HollerError.invalidVoice(voice, available: voices)
                    }

                    let sampleRate = await actor.sampleRate
                    let retry = RetryController(config: config)
                    var attempt = 0

                    while true {
                        let temp = attempt == 0
                            ? config.temperature
                            : min(config.temperature + Float(attempt) * config.retryTemperatureStep, 1.0)

                        let rawSamples = try await actor.generate(
                            text: text,
                            voice: voice,
                            config: config,
                            temperature: temp
                        )

                        let result = GenerationSession.process(
                            rawSamples: rawSamples,
                            config: config,
                            sampleRate: sampleRate
                        )

                        let decision = retry.evaluate(
                            attempt: attempt,
                            baseTemperature: config.temperature,
                            totalSamples: result.samples.count,
                            sampleRate: sampleRate,
                            wordCount: wordCount
                        )

                        switch decision {
                        case .accept:
                            if !result.samples.isEmpty {
                                continuation.yield(HollerAudioChunk(samples: result.samples, sampleRate: sampleRate))
                            }
                            continuation.finish()
                            return
                        case .retry:
                            attempt += 1
                            continue
                        case .giveUp:
                            if !result.samples.isEmpty {
                                continuation.yield(HollerAudioChunk(samples: result.samples, sampleRate: sampleRate))
                                continuation.finish()
                            } else {
                                continuation.finish(throwing: HollerError.allRetriesFailed(attempts: attempt + 1))
                            }
                            return
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Synthesize complete audio (collects all streaming chunks).
    public func synthesize(_ text: String, voice: String) async throws -> HollerAudio {
        var allSamples: [Float] = []
        let sr = await actor.sampleRate

        for try await chunk in stream(text, voice: voice) {
            allSamples.append(contentsOf: chunk.samples)
        }

        return HollerAudio(samples: allSamples, sampleRate: sr)
    }

    /// Release model and GPU memory.
    public func unload() async {
        await actor.unloadModel()
    }
}
