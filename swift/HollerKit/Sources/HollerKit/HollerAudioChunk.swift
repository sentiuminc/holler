public struct HollerAudioChunk: Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = 24000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    public var duration: Double {
        Double(samples.count) / Double(sampleRate)
    }
}

public struct HollerAudio: Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = 24000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    public var duration: Double {
        Double(samples.count) / Double(sampleRate)
    }
}
