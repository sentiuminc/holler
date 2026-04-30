public enum HollerError: Error, Sendable {
    case modelNotLoaded
    case invalidVoice(String, available: [String])
    case generationFailed(String)
    case allRetriesFailed(attempts: Int)
}
