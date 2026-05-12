import Foundation

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

    // Streaming AGC state
    private let targetRMS: Float?
    private var smoothedGain: Float = 1.0
    private let gainSmoothing: Float = 0.3

    private var speechStarted: Bool
    private var needsPause: Bool
    private var silentChunkCount = 0
    private var pendingSilence: [[Float]] = []
    private(set) var aborted = false
    private(set) var totalSamplesYielded = 0

    init(config: HollerConfiguration, sampleRate: Int = 24000, isCarryover: Bool = false, voice: String = "") {
        self.config = config
        self.sampleRate = sampleRate
        self.speechStarted = isCarryover
        self.needsPause = isCarryover
        if let lufs = config.targetLUFS {
            self.targetRMS = pow(10.0, (lufs + 0.691) / 20.0)
        } else {
            self.targetRMS = nil
        }
        self.silentAbortChunks = max(1, config.silentAbortTokens / config.streamingChunkTokens)
    }

    /// Process an incoming audio chunk. Returns zero or more chunks to yield to the caller.
    mutating func processChunk(_ samples: [Float]) -> [[Float]] {
        guard !aborted, !samples.isEmpty else { return [] }

        let chunks: [[Float]]
        if !speechStarted {
            chunks = handlePreSpeech(samples)
        } else {
            chunks = handlePostSpeech(samples)
        }

        guard targetRMS != nil else { return chunks }
        return chunks.map { normalizeChunk($0) }
    }

    private mutating func normalizeChunk(_ samples: [Float]) -> [Float] {
        guard let target = targetRMS else { return samples }

        let sumSq = samples.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = sqrt(sumSq / max(Float(samples.count), 1))

        guard rms > 0.001 else { return samples }

        let desiredGain = target / rms
        let clampedGain = min(desiredGain, 4.0)
        smoothedGain = gainSmoothing * clampedGain + (1.0 - gainSmoothing) * smoothedGain

        // Apply gain then enforce peak ceiling at 0.9
        var result = samples.map { $0 * smoothedGain }
        let resultPeak = result.map { abs($0) }.max() ?? 0
        if resultPeak > 0.9 {
            let scale = 0.9 / resultPeak
            result = result.map { $0 * scale }
            smoothedGain *= scale
        }
        return result
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
