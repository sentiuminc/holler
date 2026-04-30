/// Chunk-by-chunk streaming silence pipeline.
/// Port of `_run_generation()` from server.py, operating on audio chunks
/// as they arrive from `generateStream()`.
///
/// Pipeline:
/// 1. Skip silent chunks before speech (codec warmup removal)
/// 2. Sample-level onset trim in first speech chunk (150ms pre-roll)
/// 3. Buffer post-speech silence; abort after `silentAbortTokens`
/// 4. Hold-one-back: retain latest chunk so we can apply fadeout on stream end
/// 5. 20ms linear fade-out on final chunk
struct StreamingPipeline {
    private let config: HollerConfiguration
    private let sampleRate: Int

    private var speechStarted = false
    private var silentTokenCount = 0
    private var pendingSilence: [[Float]] = []
    private var heldChunk: [Float]?
    private(set) var aborted = false
    private(set) var totalSamplesYielded = 0

    init(config: HollerConfiguration, sampleRate: Int = 24000) {
        self.config = config
        self.sampleRate = sampleRate
    }

    /// Process an incoming audio chunk. Returns zero or more chunks to yield to the caller.
    mutating func processChunk(_ samples: [Float]) -> [[Float]] {
        guard !aborted, !samples.isEmpty else { return [] }

        if !speechStarted {
            return handlePreSpeech(samples)
        } else {
            return handlePostSpeech(samples)
        }
    }

    /// Call when the stream ends. Returns the final chunk with fadeout applied, or nil if aborted/empty.
    mutating func finish() -> [Float]? {
        guard !aborted, let held = heldChunk else { return nil }
        heldChunk = nil
        let faded = AudioPostProcessor.applyFadeOut(held, fadeMs: config.fadeOutMs, sampleRate: sampleRate)
        totalSamplesYielded += faded.count
        return faded
    }

    // MARK: - Pre-speech: waiting for first speech chunk

    private mutating func handlePreSpeech(_ samples: [Float]) -> [[Float]] {
        guard let onset = SilenceAnalyzer.findSpeechOnset(
            samples,
            threshold: config.speechOnsetThresholdRMS,
            preRollMs: config.speechOnsetPreRollMs,
            sampleRate: sampleRate
        ) else {
            silentTokenCount += 1
            if silentTokenCount >= config.silentAbortTokens {
                aborted = true
            }
            return []
        }

        speechStarted = true
        silentTokenCount = 0
        let trimmed = Array(samples[onset...])
        heldChunk = trimmed
        return []
    }

    // MARK: - Post-speech: yielding chunks, tracking silence

    private mutating func handlePostSpeech(_ samples: [Float]) -> [[Float]] {
        var output: [[Float]] = []

        if SilenceAnalyzer.hasSpeech(samples, threshold: config.speechOnsetThresholdRMS, sampleRate: sampleRate) {
            if let held = heldChunk {
                output.append(held)
                totalSamplesYielded += held.count
            }
            for pending in pendingSilence {
                output.append(pending)
                totalSamplesYielded += pending.count
            }
            pendingSilence.removeAll()
            silentTokenCount = 0
            heldChunk = samples
        } else {
            pendingSilence.append(samples)
            silentTokenCount += 1
            if silentTokenCount >= config.silentAbortTokens {
                pendingSilence.removeAll()
                aborted = true
            }
        }

        return output
    }
}
