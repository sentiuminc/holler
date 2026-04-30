import Testing
@testable import HollerKit

@Suite("SentenceBuffer")
struct SentenceBufferTests {

    @Test("Complete sentence yields immediately")
    func completeSentence() {
        var buf = SentenceBuffer()
        let sentences = buf.append("Hello world. ")
        #expect(sentences == ["Hello world."])
    }

    @Test("Partial sentence waits for boundary")
    func partialSentence() {
        var buf = SentenceBuffer()
        let s1 = buf.append("Hello ")
        #expect(s1.isEmpty)
        let s2 = buf.append("world. ")
        #expect(s2 == ["Hello world."])
    }

    @Test("Multiple sentences in one append")
    func multipleSentences() {
        var buf = SentenceBuffer()
        let sentences = buf.append("First. Second! Third? ")
        #expect(sentences.count == 3)
        #expect(sentences[0] == "First.")
        #expect(sentences[1] == "Second!")
        #expect(sentences[2] == "Third?")
    }

    @Test("Token-by-token LLM simulation")
    func tokenByToken() {
        var buf = SentenceBuffer()
        var all: [String] = []
        for token in ["Sure", ".", " Let", " me", " check", " that", ".", " "] {
            all.append(contentsOf: buf.append(token))
        }
        #expect(all == ["Sure.", "Let me check that."])
    }

    @Test("Abbreviation Dr. does not split")
    func abbreviationDr() {
        var buf = SentenceBuffer()
        let s = buf.append("Dr. Smith is here. ")
        #expect(s == ["Dr. Smith is here."])
    }

    @Test("Abbreviation etc. does not split")
    func abbreviationEtc() {
        var buf = SentenceBuffer()
        let s = buf.append("Cats, dogs, etc. are animals. ")
        #expect(s == ["Cats, dogs, etc. are animals."])
    }

    @Test("Decimal number 3.5 does not split")
    func decimalNumber() {
        var buf = SentenceBuffer()
        let s = buf.append("The value is 3.5 seconds. ")
        #expect(s == ["The value is 3.5 seconds."])
    }

    @Test("Exclamation mark splits")
    func exclamationMark() {
        var buf = SentenceBuffer()
        let s = buf.append("Hey! How are you? ")
        #expect(s.count == 2)
        #expect(s[0] == "Hey!")
        #expect(s[1] == "How are you?")
    }

    @Test("Flush returns remaining text")
    func flush() {
        var buf = SentenceBuffer()
        _ = buf.append("No period here")
        let remaining = buf.flush()
        #expect(remaining == "No period here")
        #expect(buf.isEmpty)
    }

    @Test("Flush returns nil when empty")
    func flushEmpty() {
        var buf = SentenceBuffer()
        #expect(buf.flush() == nil)
    }

    @Test("Force yield after word limit")
    func forceYield() {
        var buf = SentenceBuffer(forceYieldWordCount: 5)
        let s = buf.append("one two three four five six ")
        #expect(s.count == 1)
        #expect(s[0] == "one two three four five six")
    }

    @Test("Newline acts as boundary")
    func newlineBoundary() {
        var buf = SentenceBuffer()
        let s = buf.append("First line\nSecond line. ")
        #expect(s.count == 2)
        #expect(s[0] == "First line")
        #expect(s[1] == "Second line.")
    }

    @Test("Half sentences accumulate then yield")
    func halfSentences() {
        var buf = SentenceBuffer()
        let s1 = buf.append("I think that ")
        #expect(s1.isEmpty)
        let s2 = buf.append("you are right. And ")
        #expect(s2 == ["I think that you are right."])
        let s3 = buf.append("I agree. ")
        #expect(s3 == ["And I agree."])
    }

    @Test("1.5 sentences from LLM chunk")
    func oneAndHalfSentences() {
        var buf = SentenceBuffer()
        let s = buf.append("Sure thing. Let me ")
        #expect(s == ["Sure thing."])
        #expect(!buf.isEmpty)
        let s2 = buf.append("check. ")
        #expect(s2 == ["Let me check."])
    }

    @Test("Period at end of buffer without trailing space waits")
    func periodAtEndNoSpace() {
        var buf = SentenceBuffer()
        let s = buf.append("Hello world.")
        #expect(s == ["Hello world."])
    }

    @Test("Multiple abbreviations in sequence")
    func multipleAbbreviations() {
        var buf = SentenceBuffer()
        let s = buf.append("Dr. Mr. Smith arrived. ")
        #expect(s == ["Dr. Mr. Smith arrived."])
    }
}
