import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

actor InferenceActor {
    private var model: Qwen3TTSModel?
    private(set) var voices: [String] = []
    private(set) var sampleRate: Int = 24000

    var isLoaded: Bool { model != nil }

    func loadModel(repo: String) async throws {
        let loaded = try await TTS.loadModel(modelRepo: repo)
        guard let qwen3 = loaded as? Qwen3TTSModel else {
            throw HollerError.generationFailed("Model is not Qwen3-TTS")
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

    /// Generate complete audio within the actor's isolation.
    /// Uses the batch (non-streaming) API which is fully synchronous
    /// and safe for consecutive calls.
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

    /// Extract voice names from the HF cache config.json.
    private static func extractVoices(repo: String) -> [String] {
        let slug = repo.replacingOccurrences(of: "/", with: "_")
        let cachePath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/mlx-audio/\(slug)/config.json")

        guard let data = try? Data(contentsOf: cachePath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let talkerConfig = json["talker_config"] as? [String: Any],
              let spkId = talkerConfig["spk_id"] as? [String: Any]
        else {
            return []
        }
        return spkId.keys.sorted()
    }
}
