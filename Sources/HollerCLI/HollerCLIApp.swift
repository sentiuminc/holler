import Foundation
import HollerKit

@main
struct HollerCLI {
    static func main() async throws {
        let args = CommandLine.arguments
        var it = args.dropFirst().makeIterator()

        var text: String?
        var voice = "kit"
        var model = "sentiuminc/holler-0.6b"
        var output = "output.wav"
        var codebooks = 16
        var temperature: Float = 0.7
        var topK = 50
        var maxTokens = 500
        var noRetry = false
        var noSilenceTrim = false
        var session = false
        var tokenDelayMs: UInt64 = 15
        var benchmark = false
        var debug = false
        var talk = false

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
            case "--6bit":
                model = "sentiuminc/holler-0.6b-6bit"
            case "--no-retry":
                noRetry = true
            case "--no-silence-trim":
                noSilenceTrim = true
            case "--benchmark":
                benchmark = true
            case "--session":
                session = true
            case "--token-delay":
                guard let v = it.next(), let n = UInt64(v) else {
                    exitError("--token-delay requires milliseconds")
                }
                tokenDelayMs = n
            case "--debug":
                debug = true
            case "--talk":
                talk = true
            case "--help", "-h":
                printUsage()
                return
            default:
                exitError("Unknown argument: \(arg)")
            }
        }

        if !benchmark && text == nil {
            exitError("--text or --benchmark required. Use --help for usage.")
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
        if debug {
            config.log = { print($0) }
        }

        print("[holler] Loading model \(model)...")
        let loadStart = Date()
        let hollerModel = try await HollerModel.load(repo: model, configuration: config)
        let loadTime = Date().timeIntervalSince(loadStart)
        let voices = await hollerModel.voices
        print("[holler] Ready in \(String(format: "%.1f", loadTime))s — voices: \(voices)")

        let effectiveOutput = talk
            ? NSTemporaryDirectory() + "holler-\(ProcessInfo.processInfo.processIdentifier).wav"
            : output

        if talk {
            fputs("[holler] Note: --talk plays audio after generation finishes, not in real time.\n", stderr)
        }

        if benchmark {
            try await runBenchmark(model: hollerModel, voice: voice)
        } else if session {
            try await runSession(model: hollerModel, text: text!, voice: voice, output: effectiveOutput, tokenDelayMs: tokenDelayMs)
        } else {
            try await runSynthesize(model: hollerModel, text: text!, voice: voice, output: effectiveOutput)
        }

        if talk {
            try playAudio(path: effectiveOutput)
            try? FileManager.default.removeItem(atPath: effectiveOutput)
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

    static func runSession(model: HollerModel, text: String, voice: String, output: String, tokenDelayMs: UInt64) async throws {
        let words = text.split(separator: " ").map { String($0) }
        print("[holler] Session: \(words.count) words, \(tokenDelayMs)ms/token delay")
        print("[holler] Text: \"\(text)\"")

        let sess = model.makeSession(voice: voice)
        let sampleRate = await model.sampleRate
        let t0 = Date()
        let stats = SessionStats()

        let audioTask = Task {
            var genStart: Date?
            for try await chunk in sess.audio {
                if genStart == nil {
                    genStart = sess.generationStartDate ?? t0
                }
                _ = await stats.recordChunk(chunk.samples, t0: genStart!)
            }
        }

        for (i, word) in words.enumerated() {
            let token = (i < words.count - 1) ? word + " " : word
            sess.feed(token)
            if tokenDelayMs > 0 {
                try await Task.sleep(nanoseconds: tokenDelayMs * 1_000_000)
            }
        }
        await sess.finish()
        try await audioTask.value

        let totalMs = Date().timeIntervalSince(t0) * 1000
        let allSamples = await stats.allSamples
        let chunkCount = await stats.chunkCount
        let ttfa = await stats.ttfa
        let audioS = Double(allSamples.count) / Double(sampleRate)
        print("[holler] \(String(format: "%.1f", audioS))s audio, \(chunkCount) chunks, "
              + "TTFA=\(ttfa.map { String(format: "%.0f", $0) } ?? "—")ms, "
              + "Total=\(String(format: "%.0f", totalMs))ms")

        try writeWAV(samples: allSamples, sampleRate: sampleRate, path: output)
        print("[holler] Saved to \(output)")
    }

    static func runBenchmark(model: HollerModel, voice: String) async throws {
        let sentences = [
            "Okay so I checked and the meeting got moved to three thirty tomorrow.",
            "The file you were looking for is in your Downloads folder, not your Desktop.",
            "I found three options that fit your budget, want me to walk you through them?",
            "Your flight lands at four fifteen and there is a shuttle that runs every twenty minutes from the terminal.",
            "I compared both and honestly the second option has way better reviews, I would go with that one if the budget allows.",
            "Sure, I can look into that for you. Give me a second and I will have an answer ready.",
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
        print("Target RTF <= 1.00: \(avgRTF <= 1.00 ? "PASS" : "FAIL")")
    }

    private actor SessionStats {
        var allSamples: [Float] = []
        var chunkCount = 0
        var ttfa: Double?

        func recordChunk(_ samples: [Float], t0: Date) -> Int {
            if ttfa == nil {
                ttfa = Date().timeIntervalSince(t0) * 1000
            }
            chunkCount += 1
            allSamples.append(contentsOf: samples)
            return chunkCount
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
          holler --text "Hello world"                   Synthesize to output.wav
          holler --text "Hello world" --talk             Synthesize and play through speakers
          holler --session --text "Long paragraph."      LLM streaming simulation
          holler --benchmark                             Run 6-sentence benchmark

        Options:
          --text, -t <string>         Text to synthesize
          --talk                      Play audio through speakers instead of saving
          --session                   LLM streaming simulation (token-by-token feed)
          --benchmark                 Run 6-sentence streaming benchmark
          --voice, -v <name>          Voice name (default: kit)
          --model, -m <path-or-repo>  Model path or HF repo (default: sentiuminc/holler-0.6b)
          --6bit                      Use 6-bit quantized model (sentiuminc/holler-0.6b-6bit)
          --output, -o <path>         Output WAV path (default: output.wav)
          --codebooks <int>           Number of codebooks 1-16 (default: 16)
          --temperature <float>       Sampling temperature (default: 0.7)
          --top-k <int>               Top-k sampling (default: 50)
          --max-tokens <int>          Maximum tokens (default: 500)
          --token-delay <ms>          Delay between tokens in session mode (default: 15)
          --no-retry                  Disable retry logic
          --no-silence-trim           Disable silence trimming
          --debug                     Enable verbose debug logging
          --help, -h                  Show help
        """)
    }

    static func playAudio(path: String) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        process.arguments = [path]
        try process.run()
        process.waitUntilExit()
    }

    static func exitError(_ message: String) -> Never {
        fputs("Error: \(message)\n", stderr)
        Foundation.exit(1)
    }
}
