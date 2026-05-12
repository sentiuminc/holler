import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

public struct StreamingResult: Sendable {
    public let chunks: [[Float]]
    public let totalSamples: Int
    public let ttfaMs: Double
}

actor InferenceActor {
    private var model: Qwen3TTSModel?
    private(set) var voices: [String] = []
    private(set) var sampleRate: Int = 24000

    var isLoaded: Bool { model != nil }

    func loadModel(repo: String) async throws {
        let localURL = URL(fileURLWithPath: repo).standardizedFileURL
        let isLocal = FileManager.default.fileExists(atPath: localURL.path)

        let qwen3: Qwen3TTSModel
        if isLocal {
            qwen3 = try await Qwen3TTSModel.fromModelDirectory(localURL)
        } else {
            let loaded = try await TTS.loadModel(modelRepo: repo)
            guard let q = loaded as? Qwen3TTSModel else {
                throw HollerError.generationFailed("Model is not Qwen3-TTS")
            }
            qwen3 = q
        }
        model = qwen3
        sampleRate = qwen3.sampleRate
        voices = Self.extractVoices(repo: repo)
    }

    func unloadModel() {
        model = nil
        voices = []
        Memory.clearCache()
    }

    func generate(
        text: String,
        voice: String,
        config: HollerConfiguration,
        temperature: Float
    ) async throws -> [Float] {
        guard let model else {
            throw HollerError.modelNotLoaded
        }

        let params = GenerateParameters(
            maxTokens: config.maxTokens,
            temperature: temperature,
            topK: config.topK
        )

        let result = try await model.generate(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: "english",
            generationParameters: params,
            codebooks: config.codebooks
        )

        return result.asArray(Float.self)
    }

    func generateStreaming(
        text: String,
        voice: String,
        config: HollerConfiguration,
        temperature: Float
    ) async throws -> StreamingResult {
        guard let model else {
            throw HollerError.modelNotLoaded
        }

        let params = GenerateParameters(
            maxTokens: config.maxTokens,
            temperature: temperature,
            topK: config.topK
        )

        let t0 = Date()
        var chunks: [[Float]] = []
        var ttfa: Double?
        var totalSamples = 0

        let stream = model.generateStream(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: "english",
            generationParameters: params,
            streamingInterval: Double(config.streamingChunkTokens),
            codebooks: config.codebooks
        )

        for try await event in stream {
            switch event {
            case .audio(let arr):
                let samples = arr.asArray(Float.self)
                if ttfa == nil {
                    ttfa = Date().timeIntervalSince(t0) * 1000
                }
                chunks.append(samples)
                totalSamples += samples.count
            case .token, .info:
                break
            }
        }

        return StreamingResult(
            chunks: chunks,
            totalSamples: totalSamples,
            ttfaMs: ttfa ?? Date().timeIntervalSince(t0) * 1000
        )
    }

    /// Stream generation with silence pipeline and KV cache carryover.
    func generateStreamProcessed(
        text: String,
        voice: String,
        config: HollerConfiguration,
        temperature: Float,
        cacheState: Qwen3TTSModel.TalkerCacheState?,
        resetDecoder: Bool,
        yield yieldChunk: (HollerAudioChunk) -> Void
    ) async throws -> Bool {
        guard let model else {
            throw HollerError.modelNotLoaded
        }

        let params = GenerateParameters(
            maxTokens: config.maxTokens,
            temperature: temperature,
            topK: config.topK
        )

        var pipeline = StreamingPipeline(config: config, sampleRate: sampleRate, isCarryover: !resetDecoder, voice: voice)
        let log = config.log

        let stream = model.generateStream(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: "english",
            generationParameters: params,
            streamingInterval: Double(config.streamingChunkTokens),
            codebooks: config.codebooks,
            cacheState: cacheState,
            resetDecoder: resetDecoder
        )

        let offset = cacheState?.cache?.first?.offset ?? 0
        if offset > 0 {
            log?("[generate] KV cache carry-over: \(offset) tokens, resetDecoder=\(resetDecoder)")
        } else {
            log?("[generate] fresh generation, resetDecoder=\(resetDecoder)")
        }
        let genStart = Date()
        var rawChunkIndex = 0
        var firstYieldLogged = false

        for try await event in stream {
            if case .audio(let arr) = event {
                let samples = arr.asArray(Float.self)
                rawChunkIndex += 1
                if log != nil {
                    let elapsed = Date().timeIntervalSince(genStart) * 1000
                    let rms = Self.rms(samples)
                    log?("[generate] raw chunk \(rawChunkIndex): \(String(format: "%.0f", elapsed))ms, "
                        + "\(samples.count) samples, rms=\(String(format: "%.4f", rms))")
                }

                let processed = pipeline.processChunk(samples)
                for chunk in processed {
                    if !firstYieldLogged {
                        let yieldElapsed = Date().timeIntervalSince(genStart) * 1000
                        log?("[generate] FIRST YIELD at \(String(format: "%.0f", yieldElapsed))ms "
                            + "(after \(rawChunkIndex) raw chunks)")
                        firstYieldLogged = true
                    }
                    yieldChunk(HollerAudioChunk(samples: chunk, sampleRate: sampleRate))
                }
                if pipeline.aborted { break }
            }
        }

        let totalElapsed = Date().timeIntervalSince(genStart) * 1000
        log?("[generate] done: \(rawChunkIndex) raw chunks, "
            + "\(String(format: "%.0f", totalElapsed))ms total, "
            + "aborted=\(pipeline.aborted)")

        return pipeline.aborted
    }

    private static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumSq = samples.reduce(Float(0)) { $0 + $1 * $1 }
        return (sumSq / Float(samples.count)).squareRoot()
    }

    /// Extract voice names from config.json (local path or HF cache).
    private static func extractVoices(repo: String) -> [String] {
        let configURL: URL
        let localURL = URL(fileURLWithPath: repo).standardizedFileURL
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
            configURL = localURL.appendingPathComponent("config.json")
        } else {
            let slug = repo.replacingOccurrences(of: "/", with: "_")
            configURL = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub/mlx-audio/\(slug)/config.json")
        }

        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let talkerConfig = json["talker_config"] as? [String: Any],
              let spkId = talkerConfig["spk_id"] as? [String: Any]
        else {
            return []
        }
        return spkId.keys.sorted()
    }
}
