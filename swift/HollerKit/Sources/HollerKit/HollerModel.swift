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
    /// Yields chunks as they are generated (~130ms TTFA for first chunk).
    ///
    /// Retry strategy: first attempt streams chunks directly to the caller.
    /// If it aborts (no speech / too short), subsequent attempts collect all
    /// chunks first to avoid yielding audio from a failed generation.
    public func stream(
        _ text: String,
        voice: String
    ) -> AsyncThrowingStream<HollerAudioChunk, Error> {
        let config = self.configuration
        let actor = self.actor
        let wordCount = text.split(separator: " ").count

        return AsyncThrowingStream { continuation in
            let task = Task {
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

                        if attempt == 0 {
                            // First attempt: stream chunks directly for low TTFA
                            var totalSamples = 0
                            let aborted = try await actor.generateStreamProcessed(
                                text: text,
                                voice: voice,
                                config: config,
                                temperature: temp,
                                cacheState: nil,
                                resetDecoder: true
                            ) { chunk in
                                totalSamples += chunk.samples.count
                                continuation.yield(chunk)
                            }

                            if !aborted {
                                continuation.finish()
                                return
                            }

                            // First attempt aborted — if we already yielded audio,
                            // we can't un-yield it. Accept what we have or retry.
                            if totalSamples > 0 {
                                continuation.finish()
                                return
                            }
                        } else {
                            // Retry attempts: collect chunks, only yield if successful
                            var collected: [HollerAudioChunk] = []
                            _ = try await actor.generateStreamProcessed(
                                text: text,
                                voice: voice,
                                config: config,
                                temperature: temp,
                                cacheState: nil,
                                resetDecoder: true
                            ) { chunk in
                                collected.append(chunk)
                            }

                            let totalSamples = collected.reduce(0) { $0 + $1.samples.count }

                            let decision = retry.evaluate(
                                attempt: attempt,
                                baseTemperature: config.temperature,
                                totalSamples: totalSamples,
                                sampleRate: sampleRate,
                                wordCount: wordCount
                            )

                            switch decision {
                            case .accept:
                                for chunk in collected { continuation.yield(chunk) }
                                continuation.finish()
                                return
                            case .retry:
                                attempt += 1
                                continue
                            case .giveUp:
                                if !collected.isEmpty {
                                    for chunk in collected { continuation.yield(chunk) }
                                    continuation.finish()
                                } else {
                                    continuation.finish(throwing: HollerError.allRetriesFailed(attempts: attempt + 1))
                                }
                                return
                            }
                        }

                        attempt += 1
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Create a speech session for LLM integration.
    /// Feed text tokens via `session.feed()`, consume audio via `session.audio`.
    /// KV cache carries over between sentences automatically (Phase 2B).
    public func makeSession(voice: String) -> SpeechSession {
        SpeechSession(actor: actor, voice: voice, configuration: configuration)
    }

    /// Raw streaming test — bypasses silence pipeline, returns chunk-level data.
    public func streamRaw(_ text: String, voice: String) async throws -> StreamingResult {
        try await actor.generateStreaming(
            text: text,
            voice: voice,
            config: configuration,
            temperature: configuration.temperature
        )
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
