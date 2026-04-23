import Foundation
import Qwen3TTS
import AudioCommon

// Simulate real LLM responses — each is a multi-sentence response
// that arrives sentence-by-sentence with timing gaps
let responses: [(name: String, sentences: [String])] = [
    ("greeting", [
        "Hey!",
        "How's it going today?",
    ]),
    ("short_answer", [
        "Sure, let me check that for you.",
        "It looks like your meeting got moved to three PM.",
        "Want me to update your calendar?",
    ]),
    ("explanation", [
        "Great question.",
        "The weather today is going to be mostly sunny with a high of seventy-two.",
        "There's a slight chance of rain later this evening, so you might want to grab an umbrella just in case.",
        "Tomorrow looks even better though.",
    ]),
    ("technical", [
        "Okay.",
        "I found the issue.",
        "The server was returning a five hundred error because the database connection pool was exhausted.",
        "I've increased the pool size from ten to fifty and restarted the service.",
        "Everything should be back to normal now.",
    ]),
    ("single_word", [
        "Done.",
    ]),
    ("two_words", [
        "Got it.",
    ]),
    ("natural_conversation", [
        "Oh, interesting!",
        "I hadn't thought about it that way before.",
        "Let me think about it and get back to you.",
    ]),
]

let checkpoints: [(name: String, path: String, voices: [(name: String, label: String)])] = [
    ("v6-4bit", "/Users/nagy/Desktop/Files/AI/holler/checkpoints/katie-v6-4bit",
     [("katie", "Katie v6 4-bit")]),
    ("v7-4bit", "/Users/nagy/Desktop/Files/AI/holler/checkpoints/katie-joe-v7-4bit",
     [("katie", "Katie v7 4-bit"), ("joe", "Joe v7 4-bit")]),
]

let outputBase = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Downloads/holler-sentence-queue")

func runTests() async throws {
    try FileManager.default.createDirectory(at: outputBase, withIntermediateDirectories: true)

    for checkpoint in checkpoints {
        print("\n{'='}")
        print("CHECKPOINT: \(checkpoint.name)")
        print("============================================================")

        let checkpointURL = URL(fileURLWithPath: checkpoint.path)
        let loadStart = CFAbsoluteTimeGetCurrent()
        let model: Qwen3TTSModel
        do {
            model = try await Qwen3TTSModel.fromPretrained(
                modelId: checkpoint.path,
                cacheDir: checkpointURL)
            print("  Load time: \(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - loadStart))s")
            print("  Speakers: \(model.availableSpeakers)")
        } catch {
            print("  FAILED TO LOAD: \(error)")
            continue
        }

        for voice in checkpoint.voices {
            print("\n  === \(voice.label) ===")

            let voiceDir = outputBase.appendingPathComponent("\(checkpoint.name)-\(voice.name)")
            try FileManager.default.createDirectory(at: voiceDir, withIntermediateDirectories: true)

            // === TEST 1: Individual sentence timing ===
            print("\n  [Individual sentence timing]")
            print("  Measuring how long each sentence takes to generate:")
            for response in responses {
                for (i, sentence) in response.sentences.enumerated() {
                    let start = CFAbsoluteTimeGetCurrent()
                    let audio = model.synthesize(
                        text: sentence,
                        language: "english",
                        speaker: voice.name,
                        sampling: SamplingConfig(temperature: 0.6, topK: 50, maxTokens: 500))
                    let elapsed = CFAbsoluteTimeGetCurrent() - start
                    let audioDur = Double(audio.count) / 24000.0
                    let words = sentence.split(separator: " ").count

                    print("    \(String(format: "%5.0f", elapsed * 1000))ms → \(String(format: "%.2f", audioDur))s audio | \(words) words | \"\(sentence)\"")

                    let wavPath = voiceDir.appendingPathComponent("\(response.name)_\(i).wav")
                    try WAVWriter.write(samples: audio, sampleRate: 24000, to: wavPath)
                }
            }

            // === TEST 2: Sentence queue simulation ===
            print("\n  [Sentence queue — simulating ivi pattern]")
            print("  LLM streams sentences → TTS generates sequentially → measure time-to-first-audio")

            for response in responses {
                let queueStart = CFAbsoluteTimeGetCurrent()
                var firstAudioReady: Double?
                var allAudio: [[Float]] = []

                for (i, sentence) in response.sentences.enumerated() {
                    let sentStart = CFAbsoluteTimeGetCurrent()
                    let audio = model.synthesize(
                        text: sentence,
                        language: "english",
                        speaker: voice.name,
                        sampling: SamplingConfig(temperature: 0.6, topK: 50, maxTokens: 500))
                    let sentElapsed = CFAbsoluteTimeGetCurrent() - sentStart

                    allAudio.append(audio)

                    if firstAudioReady == nil {
                        firstAudioReady = CFAbsoluteTimeGetCurrent() - queueStart
                    }

                    let audioDur = Double(audio.count) / 24000.0
                    let cumAudioDur = allAudio.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
                    let wallTime = CFAbsoluteTimeGetCurrent() - queueStart

                    // Is generation keeping ahead of playback?
                    let ahead = cumAudioDur - wallTime
                    let status = ahead > 0 ? "ahead by \(String(format: "%.2f", ahead))s" : "BEHIND by \(String(format: "%.2f", -ahead))s"

                    if i == 0 {
                        print("    \"\(response.name)\" (\(response.sentences.count) sentences):")
                    }
                    print("      [\(i)] \(String(format: "%4.0f", sentElapsed * 1000))ms gen → \(String(format: "%.2f", audioDur))s audio | \(status)")
                }

                let totalWall = CFAbsoluteTimeGetCurrent() - queueStart
                let totalAudio = allAudio.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
                print("      TTFA: \(String(format: "%.0f", (firstAudioReady ?? 0) * 1000))ms | total: \(String(format: "%.1f", totalWall))s wall → \(String(format: "%.1f", totalAudio))s audio")

                // Save concatenated audio
                let concat = allAudio.flatMap { $0 }
                let wavPath = voiceDir.appendingPathComponent("queue_\(response.name).wav")
                try WAVWriter.write(samples: concat, sampleRate: 24000, to: wavPath)
            }
        }
    }

    print("\n\nAll audio saved to: \(outputBase.path)")
}

do {
    try await runTests()
} catch {
    print("Fatal error: \(error)")
    exit(1)
}
