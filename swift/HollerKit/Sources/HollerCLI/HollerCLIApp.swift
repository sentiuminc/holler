import Foundation
import HollerKit

@main
struct HollerCLI {
    static func main() async throws {
        let args = CommandLine.arguments
        var it = args.dropFirst().makeIterator()

        var text: String?
        var voice = "kit"
        var model = "sentium/holler-0.6b-6bit"
        var output = "output.wav"
        var codebooks = 12
        var temperature: Float = 0.6
        var topK = 50
        var maxTokens = 500
        var noRetry = false
        var noSilenceTrim = false
        var benchmark = false
        var streamingTest = false
        var sessionTest = false

        while let arg = it.next() {
            switch arg {
            case "--text", "-t":
                guard let v = it.next() else { exitError("--text requires a value") }
                text = v
            case "--voice", "-v":
                guard let v = it.next() else { exitError("--voice requires a value") }
                voice = v
            case "--model", "-m":
                guard let v = it.next() else { exitError("--model requires a value") }
                model = v
            case "--output", "-o":
                guard let v = it.next() else { exitError("--output requires a value") }
                output = v
            case "--codebooks":
                guard let v = it.next(), let n = Int(v), (1...16).contains(n) else {
                    exitError("--codebooks requires an integer 1-16")
                }
                codebooks = n
            case "--temperature":
                guard let v = it.next(), let f = Float(v) else {
                    exitError("--temperature requires a float")
                }
                temperature = f
            case "--top-k":
                guard let v = it.next(), let n = Int(v) else {
                    exitError("--top-k requires an integer")
                }
                topK = n
            case "--max-tokens":
                guard let v = it.next(), let n = Int(v) else {
                    exitError("--max-tokens requires an integer")
                }
                maxTokens = n
            case "--no-retry":
                noRetry = true
            case "--no-silence-trim":
                noSilenceTrim = true
            case "--benchmark":
                benchmark = true
            case "--streaming-test":
                streamingTest = true
            case "--session-test":
                sessionTest = true
            case "--help", "-h":
                printUsage()
                return
            default:
                exitError("Unknown argument: \(arg)")
            }
        }

        if !benchmark && !streamingTest && !sessionTest && text == nil {
            exitError("--text, --benchmark, --streaming-test, or --session-test required. Use --help for usage.")
        }

        var config = HollerConfiguration()
        config.codebooks = codebooks
        config.temperature = temperature
        config.topK = topK
        config.maxTokens = maxTokens
        if noRetry { config.maxRetries = 0 }
        if noSilenceTrim {
            config.silentAbortTokens = 9999
            config.speechOnsetThresholdRMS = 0
        }

        print("[holler] Loading model \(model)...")
        let loadStart = Date()
        let hollerModel = try await HollerModel.load(repo: model, configuration: config)
        let loadTime = Date().timeIntervalSince(loadStart)
        let voices = await hollerModel.voices
        print("[holler] Ready in \(String(format: "%.1f", loadTime))s — voices: \(voices)")

        if sessionTest {
            try await runSessionTest(model: hollerModel, voice: voice)
        } else if streamingTest {
            try await runStreamingTest(model: hollerModel, voice: voice)
        } else if benchmark {
            try await runBenchmark(model: hollerModel, voice: voice)
        } else {
            try await runSynthesize(model: hollerModel, text: text!, voice: voice, output: output)
        }

        await hollerModel.unload()
    }

    static func runSynthesize(model: HollerModel, text: String, voice: String, output: String) async throws {
        print("[holler] Synthesizing: \"\(text)\"")
        let t0 = Date()

        let audio = try await model.synthesize(text, voice: voice)

        let totalMs = Date().timeIntervalSince(t0) * 1000
        let rtf = (totalMs / 1000) / audio.duration
        print("[holler] \(String(format: "%.1f", audio.duration))s audio, "
              + "\(String(format: "%.0f", totalMs))ms, "
              + "RTF=\(String(format: "%.3f", rtf))")

        try writeWAV(samples: audio.samples, sampleRate: audio.sampleRate, path: output)
        print("[holler] Saved to \(output)")
    }

    static func runBenchmark(model: HollerModel, voice: String) async throws {
        let sentences = [
            "Hey!",
            "Got it.",
            "What is on your mind?",
            "Yeah, that is pretty common with voice input.",
            "Something about the umami thing appeals to me.",
            "Hold on, speak a couple sentences, release, and let me know if the gap is gone.",
        ]

        print("[holler] Benchmark (\(sentences.count) sentences)")
        print(String(repeating: "=", count: 70))
        let header = "Text".padding(toLength: 45, withPad: " ", startingAt: 0)
            + "  TTFA   Total  Audio    RTF"
        print(header)
        print(String(repeating: "-", count: 70))

        var totalRTF: Double = 0
        var totalTTFA: Double = 0

        for text in sentences {
            let t0 = Date()
            var ttfa: Double?
            var totalSamples = 0

            for try await chunk in model.stream(text, voice: voice) {
                if ttfa == nil {
                    ttfa = Date().timeIntervalSince(t0) * 1000
                }
                totalSamples += chunk.samples.count
            }

            let totalMs = Date().timeIntervalSince(t0) * 1000
            let sampleRate = await model.sampleRate
            let audioS = Double(totalSamples) / Double(sampleRate)
            let rtf = audioS > 0 ? (totalMs / 1000) / audioS : 999

            let ttfaStr = ttfa.map { String(format: "%.0f", $0) + "ms" } ?? "—"
            let display = String(text.prefix(44)).padding(toLength: 45, withPad: " ", startingAt: 0)
            let line = "\(display) \(ttfaStr.padding(toLength: 6, withPad: " ", startingAt: 0))"
                + " \(String(format: "%.0f", totalMs).padding(toLength: 5, withPad: " ", startingAt: 0))ms"
                + " \(String(format: "%.1f", audioS).padding(toLength: 5, withPad: " ", startingAt: 0))s"
                + " \(String(format: "%.3f", rtf))"
            print(line)

            totalRTF += rtf
            totalTTFA += ttfa ?? 0
        }

        let avgRTF = totalRTF / Double(sentences.count)
        let avgTTFA = totalTTFA / Double(sentences.count)
        print("")
        print("Avg RTF: \(String(format: "%.3f", avgRTF)), Avg TTFA: \(String(format: "%.0f", avgTTFA))ms")
        print("Target RTF <= 0.50: \(avgRTF <= 0.50 ? "PASS" : "FAIL")")
    }

    static func runSessionTest(model: HollerModel, voice: String) async throws {
        let tokens = [
            "Sure", ".", " Let", " me", " check", " that", " for", " you", ".",
            " I", " think", " the", " answer", " is", " forty", " two", ".",
            " Does", " that", " help", "?",
        ]

        print("[session-test] Simulating LLM token stream (\(tokens.count) tokens)")
        print("[session-test] Text: \(tokens.joined())")
        print(String(repeating: "=", count: 70))

        let session = model.makeSession(voice: voice)
        let sampleRate = await model.sampleRate
        let t0 = Date()

        let stats = SessionStats()

        let audioTask = Task {
            for try await chunk in session.audio {
                let count = await stats.recordChunk(chunk.samples.count, t0: t0)
                let audioMs = Double(chunk.samples.count) / Double(sampleRate) * 1000
                print("[session-test] Chunk \(count): \(chunk.samples.count) samples (\(String(format: "%.0f", audioMs))ms audio)")
            }
        }

        for token in tokens {
            session.feed(token)
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        await session.finish()

        try await audioTask.value

        let totalMs = Date().timeIntervalSince(t0) * 1000
        let totalSamples = await stats.totalSamples
        let chunkCount = await stats.chunkCount
        let ttfa = await stats.ttfa
        let audioS = Double(totalSamples) / Double(sampleRate)
        print(String(repeating: "=", count: 70))
        print("[session-test] \(chunkCount) chunks, \(totalSamples) samples (\(String(format: "%.1f", audioS))s audio)")
        print("[session-test] TTFA: \(ttfa.map { String(format: "%.0f", $0) } ?? "—")ms, Total: \(String(format: "%.0f", totalMs))ms")
        print("[session-test] Result: \(totalSamples > 0 ? "PASSED" : "FAILED")")
    }

    private actor SessionStats {
        var totalSamples = 0
        var chunkCount = 0
        var ttfa: Double?

        func recordChunk(_ samples: Int, t0: Date) -> Int {
            if ttfa == nil {
                ttfa = Date().timeIntervalSince(t0) * 1000
            }
            chunkCount += 1
            totalSamples += samples
            return chunkCount
        }
    }

    static func runStreamingTest(model: HollerModel, voice: String) async throws {
        let sentences = [
            "Hey!",
            "Got it.",
            "What is on your mind?",
            "Yeah, that is pretty common with voice input.",
            "Something about the umami thing appeals to me.",
            "Hold on, speak a couple sentences, release, and let me know if the gap is gone.",
        ]

        print("[streaming-test] Testing generateStream() with \(sentences.count) consecutive calls")
        print("[streaming-test] This tests whether streaming works safely for back-to-back generation")
        print(String(repeating: "=", count: 80))

        let header = "# ".padding(toLength: 3, withPad: " ", startingAt: 0)
            + "Text".padding(toLength: 42, withPad: " ", startingAt: 0)
            + "  TTFA   Total  Chunks  Samples  Status"
        print(header)
        print(String(repeating: "-", count: 80))

        var allPassed = true

        for (i, text) in sentences.enumerated() {
            let t0 = Date()
            var status = "OK"

            do {
                let result = try await model.streamRaw(text, voice: voice)
                let totalMs = Date().timeIntervalSince(t0) * 1000
                let sampleRate = await model.sampleRate

                let allSamples = result.chunks.flatMap { $0 }
                let savePath = NSString(string: "~/Downloads/holler-streaming-test/\(String(format: "%02d", i + 1))-\(voice).wav").expandingTildeInPath
                try writeWAV(samples: allSamples, sampleRate: sampleRate, path: savePath)

                let display = String(text.prefix(41)).padding(toLength: 42, withPad: " ", startingAt: 0)
                let ttfaStr = String(format: "%.0fms", result.ttfaMs).padding(toLength: 6, withPad: " ", startingAt: 0)
                let totalStr = String(format: "%.0fms", totalMs).padding(toLength: 6, withPad: " ", startingAt: 0)
                let chunksStr = String(result.chunks.count).padding(toLength: 6, withPad: " ", startingAt: 0)
                let samplesStr = String(result.totalSamples).padding(toLength: 7, withPad: " ", startingAt: 0)
                let audioMs = Double(result.totalSamples) / Double(sampleRate) * 1000

                if result.totalSamples == 0 {
                    status = "EMPTY"
                    allPassed = false
                } else if audioMs < 100 {
                    status = "SHORT"
                    allPassed = false
                }

                print("\(String(i + 1).padding(toLength: 3, withPad: " ", startingAt: 0))"
                    + "\(display) \(ttfaStr) \(totalStr) \(chunksStr) \(samplesStr)  \(status)")
            } catch {
                status = "CRASH"
                allPassed = false
                let display = String(text.prefix(41)).padding(toLength: 42, withPad: " ", startingAt: 0)
                print("\(String(i + 1).padding(toLength: 3, withPad: " ", startingAt: 0))"
                    + "\(display)  —      —      —       —        \(status): \(error)")
            }
        }

        print(String(repeating: "=", count: 80))
        print("[streaming-test] Result: \(allPassed ? "ALL PASSED" : "FAILURES DETECTED")")
        if allPassed {
            print("[streaming-test] Streaming consecutive calls work safely!")
            print("[streaming-test] Phase 2A can proceed with generateStream()")
        }
    }

    static func writeWAV(samples: [Float], sampleRate: Int, path: String) throws {
        var data = Data()
        let pcm16 = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * 32767)
        }
        let dataSize = UInt32(pcm16.count * 2)
        let fileSize = UInt32(36 + dataSize)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) }) // PCM
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) }) // mono
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) }) // byte rate
        data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) }) // block align
        data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) }) // bits per sample

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })
        for sample in pcm16 {
            data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        try data.write(to: URL(fileURLWithPath: path))
    }

    static func printUsage() {
        print("""
        holler — Production-quality TTS for Holler voices

        Usage:
          holler --text "Hello world" --voice kit [options]
          holler --benchmark [options]

        Options:
          --text, -t <string>         Text to synthesize (required unless --benchmark)
          --voice, -v <name>          Voice name (default: kit)
          --model, -m <path-or-repo>  Model path or HF repo (default: sentium/holler-0.6b-6bit)
          --output, -o <path>         Output WAV path (default: output.wav)
          --codebooks <int>           Number of codebooks 1-16 (default: 12)
          --temperature <float>       Sampling temperature (default: 0.6)
          --top-k <int>               Top-k sampling (default: 50)
          --max-tokens <int>          Maximum tokens (default: 500)
          --no-retry                  Disable retry logic
          --no-silence-trim           Disable silence trimming
          --benchmark                 Run 6-sentence benchmark
          --help, -h                  Show help
        """)
    }

    static func exitError(_ message: String) -> Never {
        fputs("Error: \(message)\n", stderr)
        Foundation.exit(1)
    }
}
