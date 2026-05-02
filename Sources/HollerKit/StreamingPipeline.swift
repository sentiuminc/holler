/// Chunk-by-chunk streaming silence pipeline.
/// Port of `_run_generation()` from server.py, operating on audio chunks
/// as they arrive from `generateStream()`.
///
/// Pipeline:
/// 1. Skip silent chunks before speech (codec warmup removal)
/// 2. Sample-level onset trim in first speech chunk (150ms pre-roll)
/// 3. Buffer post-speech silence; abort after `silentAbortTokens`
struct StreamingPipeline {
    private let config: HollerConfiguration
    private let sampleRate: Int
    private let silentAbortChunks: Int

    private var speechStarted: Bool
    private var needsPause: Bool
    private var silentChunkCount = 0
    private var pendingSilence: [[Float]] = []
    private(set) var aborted = false
    private(set) var totalSamplesYielded = 0

    init(config: HollerConfiguration, sampleRate: Int = 24000, isCarryover: Bool = false) {
        self.config = config
        self.sampleRate = sampleRate
        self.speechStarted = isCarryover
        self.needsPause = isCarryover
        // silentAbortTokens is in codec tokens; each chunk is streamingChunkTokens tokens
        self.silentAbortChunks = max(1, config.silentAbortTokens / config.streamingChunkTokens)
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

    // MARK: - Pre-speech: waiting for first speech chunk

    private mutating func handlePreSpeech(_ samples: [Float]) -> [[Float]] {
        guard let onset = SilenceAnalyzer.findSpeechOnset(
            samples,
            threshold: config.speechOnsetThresholdRMS,
            preRollMs: config.speechOnsetPreRollMs,
            sampleRate: sampleRate
        ) else {
            silentChunkCount += 1
            if silentChunkCount >= silentAbortChunks {
                config.log?("[pipeline] pre-speech abort after \(silentChunkCount) silent chunks (~\(silentChunkCount * config.streamingChunkTokens) tokens)")
                aborted = true
            }
            return []
        }

        speechStarted = true
        silentChunkCount = 0
        let trimmed = Array(samples[onset...])
        totalSamplesYielded += trimmed.count
        config.log?("[pipeline] speech onset at sample \(onset), trimmed to \(trimmed.count) samples")
        return [trimmed]
    }

    // MARK: - Post-speech: yielding chunks, tracking silence

    private mutating func handlePostSpeech(_ samples: [Float]) -> [[Float]] {
        var output: [[Float]] = []

        if SilenceAnalyzer.hasSpeech(samples, threshold: config.speechOnsetThresholdRMS, sampleRate: sampleRate) {
            if needsPause {
                let pauseMs = Int.random(in: config.carryoverPauseMinMs...config.carryoverPauseMaxMs)
                let pauseSamples = Int(Double(sampleRate) * Double(pauseMs) / 1000.0)
                let naturalMs = pendingSilence.reduce(0) { $0 + $1.count } * 1000 / sampleRate
                config.log?("[pipeline] carryover pause: \(pauseMs)ms (replaced \(naturalMs)ms natural)")
                pendingSilence.removeAll()
                pendingSilence.append([Float](repeating: 0, count: pauseSamples))
                needsPause = false
            }

            for pending in pendingSilence {
                output.append(pending)
                totalSamplesYielded += pending.count
            }
            pendingSilence.removeAll()
            silentChunkCount = 0
            output.append(samples)
            totalSamplesYielded += samples.count
        } else {
            pendingSilence.append(samples)
            silentChunkCount += 1
            if silentChunkCount >= silentAbortChunks {
                config.log?("[pipeline] post-speech abort after \(silentChunkCount) silent chunks (~\(silentChunkCount * config.streamingChunkTokens) tokens, \(totalSamplesYielded) samples yielded)")
                pendingSilence.removeAll()
                aborted = true
            }
        }

        return output
    }
}
