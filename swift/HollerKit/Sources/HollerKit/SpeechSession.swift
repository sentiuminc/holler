import Foundation

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
        guard !cancelled else { return }

        let sentences = sentenceBuffer.append(text)
        for sentence in sentences {
            sentenceContinuation.yield(sentence)
        }
    }

    /// Signal that no more text will arrive. Flushes remaining
    /// buffered text and waits for all generation to complete.
    public func finish() async {
        guard !cancelled else { return }

        if let remaining = sentenceBuffer.flush() {
            sentenceContinuation.yield(remaining)
        }
        sentenceContinuation.finish()
        await generationTask?.value
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

    private func synthesizeSentence(_ text: String) async {
        guard !cancelled else { return }

        let wordCount = text.split(separator: " ").count
        let retry = RetryController(config: config)
        var attempt = 0

        while !cancelled {
            let temp = attempt == 0
                ? config.temperature
                : min(config.temperature + Float(attempt) * config.retryTemperatureStep, 1.0)

            do {
                var totalSamples = 0
                let aborted = try await actor.generateStreamProcessed(
                    text: text,
                    voice: voice,
                    config: config,
                    temperature: temp
                ) { [weak self] chunk in
                    guard let self, !self.cancelled else { return }
                    totalSamples += chunk.samples.count
                    self.audioContinuation.yield(chunk)
                }

                if !aborted || cancelled {
                    return
                }

                if totalSamples > 0 { return }

                let sampleRate = await actor.sampleRate
                let decision = retry.evaluate(
                    attempt: attempt,
                    baseTemperature: config.temperature,
                    totalSamples: totalSamples,
                    sampleRate: sampleRate,
                    wordCount: wordCount
                )

                switch decision {
                case .accept:
                    return
                case .retry:
                    attempt += 1
                    continue
                case .giveUp:
                    return
                }
            } catch {
                if !cancelled {
                    audioContinuation.finish(throwing: error)
                }
                return
            }
        }
    }
}
