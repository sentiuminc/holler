import Foundation

public struct SentenceBuffer: Sendable {
    private var buffer: String = ""
    private var wordCount: Int = 0
    private let forceYieldWordCount: Int

    private static let abbreviations: Set<String> = [
        "dr", "mr", "mrs", "ms", "jr", "sr", "st", "vs",
        "etc", "prof", "gen", "gov", "sgt", "cpl", "pvt",
        "ave", "blvd", "dept", "est", "fig", "inc", "ltd",
    ]

    public init(forceYieldWordCount: Int = 30) {
        self.forceYieldWordCount = forceYieldWordCount
    }

    public mutating func append(_ text: String) -> [String] {
        buffer.append(text)
        wordCount += text.split(separator: " ").count
        return extractSentences()
    }

    public mutating func flush() -> String? {
        let remaining = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        wordCount = 0
        return remaining.isEmpty ? nil : remaining
    }

    public var isEmpty: Bool { buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    private mutating func extractSentences() -> [String] {
        var sentences: [String] = []

        while true {
            if let idx = findSentenceBoundary() {
                let endIndex = buffer.index(after: idx)
                let sentence = String(buffer[buffer.startIndex..<endIndex])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !sentence.isEmpty {
                    sentences.append(sentence)
                }
                buffer = String(buffer[endIndex...])
                wordCount = buffer.split(separator: " ").count
            } else if wordCount >= forceYieldWordCount {
                if let clauseIdx = findLastClauseBoundary() {
                    let endIndex = buffer.index(after: clauseIdx)
                    let clause = String(buffer[buffer.startIndex..<endIndex])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !clause.isEmpty {
                        sentences.append(clause)
                    }
                    buffer = String(buffer[endIndex...])
                    wordCount = buffer.split(separator: " ").count
                } else {
                    let forced = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !forced.isEmpty {
                        sentences.append(forced)
                    }
                    buffer = ""
                    wordCount = 0
                }
                break
            } else {
                break
            }
        }

        return sentences
    }

    private func findSentenceBoundary() -> String.Index? {
        var i = buffer.startIndex
        while i < buffer.endIndex {
            let ch = buffer[i]

            if ch == "!" || ch == "?" {
                let afterPunct = buffer.index(after: i)
                if afterPunct == buffer.endIndex || buffer[afterPunct].isWhitespace || buffer[afterPunct].isNewline {
                    return i
                }
            }

            if ch == "." {
                let afterDot = buffer.index(after: i)
                let followedBySpaceOrEnd = afterDot == buffer.endIndex
                    || buffer[afterDot].isWhitespace
                    || buffer[afterDot].isNewline

                if followedBySpaceOrEnd && !isAbbreviation(before: i) && !isDecimalNumber(at: i) {
                    return i
                }
            }

            if ch == "\n" {
                let trimmed = String(buffer[buffer.startIndex...i])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return i
                }
            }

            i = buffer.index(after: i)
        }
        return nil
    }

    private func isAbbreviation(before dotIndex: String.Index) -> Bool {
        let end = dotIndex
        var start = end
        while start > buffer.startIndex {
            let prev = buffer.index(before: start)
            if buffer[prev].isLetter {
                start = prev
            } else {
                break
            }
        }
        guard start < end else { return false }
        let word = String(buffer[start..<end]).lowercased()
        return Self.abbreviations.contains(word)
    }

    private func findLastClauseBoundary() -> String.Index? {
        let clauseChars: Set<Character> = [",", ";", ":", "\u{2014}"]
        var lastBoundary: String.Index? = nil
        var wordsBeforeBoundary = 0
        var currentWords = 0

        var i = buffer.startIndex
        while i < buffer.endIndex {
            let ch = buffer[i]
            if ch == " " { currentWords += 1 }
            if clauseChars.contains(ch) {
                let afterPunct = buffer.index(after: i)
                if afterPunct < buffer.endIndex && (buffer[afterPunct].isWhitespace || buffer[afterPunct].isNewline) {
                    if currentWords >= 5 {
                        lastBoundary = i
                        wordsBeforeBoundary = currentWords
                    }
                }
            }
            i = buffer.index(after: i)
        }

        guard let boundary = lastBoundary, wordsBeforeBoundary >= 5 else { return nil }
        return boundary
    }

    private func isDecimalNumber(at dotIndex: String.Index) -> Bool {
        guard dotIndex > buffer.startIndex else { return false }
        let before = buffer.index(before: dotIndex)
        guard buffer[before].isNumber else { return false }

        let after = buffer.index(after: dotIndex)
        guard after < buffer.endIndex, buffer[after].isNumber else { return false }

        return true
    }
}
