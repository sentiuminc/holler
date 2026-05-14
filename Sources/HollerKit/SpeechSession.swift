import Foundation
import MLXAudioTTS

/// Session for streaming text-to-speech with automatic sentence buffering
/// and KV cache carryover between sentences.
///
/// Usage:
/// ```swift
/// let session = model.makeSession(voice: "kit")
///
/// // Feed text and consume audio concurrently
/// async let playback: Void = {
///     for try await chunk in session.audio {
///         player.scheduleBuffer(chunk)
///     }
/// }()
///
/// for await token in llmStream {
///     await session.feed(token)
/// }
/// await session.finish()
/// try await playback
/// ```
public final class SpeechSession: @unchecked Sendable {
    private let actor: InferenceActor
    private let voice: String
    private let config: HollerConfiguration

    private var sentenceBuffer = SentenceBuffer()
    private let audioContinuation: AsyncThrowingStream<HollerAudioChunk, Error>.Continuation
    private var cancelled = false

    private let sentenceStream: AsyncStream<String>
    private let sentenceContinuation: AsyncStream<String>.Continuation
    private var generationTask: Task<Void, Never>?

    private var cacheState = Qwen3TTSModel.TalkerCacheState()
    private var isFirstSentence = true
    private var finished = false
    /// Timestamp when the first sentence started generating.
    public private(set) var generationStartDate: Date?

    /// Audio output stream. Yields chunks as sentences are synthesized.
    public let audio: AsyncThrowingStream<HollerAudioChunk, Error>

    init(actor: InferenceActor, voice: String, configuration: HollerConfiguration) {
        self.actor = actor
        self.voice = voice
        self.config = configuration

        let (audioStream, audioCont) = AsyncThrowingStream<HollerAudioChunk, Error>.makeStream()
        self.audio = audioStream
        self.audioContinuation = audioCont

        let (sentStream, sentCont) = AsyncStream<String>.makeStream()
        self.sentenceStream = sentStream
        self.sentenceContinuation = sentCont

        self.generationTask = Task { [weak self] in
            guard let self else { return }
            for await sentence in self.sentenceStream {
                guard !self.cancelled else { break }
                await self.synthesizeSentence(sentence)
            }
            if !self.cancelled {
                self.audioContinuation.finish()
            }
        }
    }

    /// Feed text tokens. Returns immediately — sentences are queued
    /// for generation in the background.
    public func feed(_ text: String) {
        if finished {
            config.log?("[session] feed after finish (dropped): \"\(text)\"")
        }
        guard !cancelled, !finished else { return }

        let sentences = sentenceBuffer.append(text)
        for sentence in sentences {
            config.log?("[session] sentence ready: \"\(sentence)\"")
            sentenceContinuation.yield(sentence)
        }
    }

    /// Signal that no more text will arrive. Flushes remaining
    /// buffered text and waits for all generation to complete.
    public func finish() async {
        guard !cancelled, !finished else { return }
        finished = true

        if let remaining = sentenceBuffer.flush() {
            config.log?("[session] flush remainder: \"\(remaining)\"")
            sentenceContinuation.yield(remaining)
        }
        sentenceContinuation.finish()
        await generationTask?.value
        config.log?("[session] finished")
    }

    /// Cancel the session. Stops generation, discards buffered text,
    /// and finishes the audio stream immediately.
    public func cancel() {
        cancelled = true
        sentenceBuffer = SentenceBuffer()
        sentenceContinuation.finish()
        generationTask?.cancel()
        audioContinuation.finish()
    }

    // MARK: - Internal

    /// Retry semantics matching Python server.py generate_audio():
    /// - First attempt: use carryover cache + decoder state
    /// - Retries: fresh cache, fresh decoder, bumped temperature
    /// - If first attempt poisons the cache, reset it before retries
    private func synthesizeSentence(_ text: String) async {
        guard !cancelled else { return }

        if generationStartDate == nil {
            generationStartDate = Date()
        }

        let resetDecoder = isFirstSentence
        let wordCount = text.split(separator: " ").count
        let retry = RetryController(config: config)
        var attempt = 0

        config.log?("[session] synthesize: \"\(text)\" (firstSentence=\(isFirstSentence), resetDecoder=\(resetDecoder))")

        while !cancelled {
            let temp = attempt == 0
                ? config.temperature
                : min(config.temperature + Float(attempt) * config.retryTemperatureStep, 1.0)

            // First attempt: use carryover. Retries: fresh state (matching Python).
            let useCache: Qwen3TTSModel.TalkerCacheState? = attempt == 0 ? cacheState : nil
            let useReset = attempt == 0 ? resetDecoder : true

            if attempt > 0 {
                config.log?("[session] retry \(attempt)/\(config.maxRetries) temp=\(String(format: "%.1f", temp)) (fresh cache+decoder)")
            }

            do {
                var totalSamples = 0
                let aborted = try await actor.generateStreamProcessed(
                    text: text,
                    voice: voice,
                    config: config,
                    temperature: temp,
                    cacheState: useCache,
                    resetDecoder: useReset
                ) { [weak self] chunk in
                    guard let self, !self.cancelled else { return }
                    totalSamples += chunk.samples.count
                    self.audioContinuation.yield(chunk)
                }

                if cancelled { return }

                if !aborted {
                    config.log?("[session] completed: \(totalSamples) samples")
                    isFirstSentence = false
                    return
                }

                if totalSamples > 0 {
                    config.log?("[session] aborted but had \(totalSamples) samples, accepting")
                    isFirstSentence = false
                    return
                }

                // First attempt aborted with no speech — cache is poisoned
                if attempt == 0 {
                    config.log?("[session] first attempt failed, resetting cache")
                    cacheState = Qwen3TTSModel.TalkerCacheState()
                }

                let sampleRate = await actor.sampleRate
                let decision = retry.evaluate(
                    attempt: attempt,
                    baseTemperature: config.temperature,
                    totalSamples: totalSamples,
                    sampleRate: sampleRate,
                    wordCount: wordCount
                )

                config.log?("[session] retry decision: \(decision) (attempt=\(attempt), samples=\(totalSamples))")

                switch decision {
                case .accept:
                    isFirstSentence = false
                    return
                case .retry:
                    attempt += 1
                    continue
                case .giveUp:
                    config.log?("[session] all retries failed for: \"\(text)\"")
                    if totalSamples > 0 {
                        isFirstSentence = false
                    } else {
                        isFirstSentence = true
                    }
                    return
                }
            } catch {
                config.log?("[session] error: \(error)")
                if !cancelled {
                    audioContinuation.finish(throwing: error)
                }
                return
            }
        }
    }
}
