"""Stress test: simulate ivi's real usage pattern.
- LLM streams sentences → each goes to TTS immediately
- Measure: per-sentence RTF, TTFA, total response latency
- Test: back-to-back requests (no gap), concurrent potential issues
"""
import time
import json
import urllib.request

SAMPLE_RATE = 24000

conversations = [
    {
        "name": "greeting",
        "sentences": ["Hey!", "How's it going today?"],
    },
    {
        "name": "short_answer",
        "sentences": [
            "Sure, let me check that for you.",
            "It looks like your meeting got moved to three PM.",
            "Want me to update your calendar?",
        ],
    },
    {
        "name": "explanation",
        "sentences": [
            "Great question.",
            "The weather today is going to be mostly sunny with a high of seventy-two.",
            "There is a slight chance of rain later this evening, so you might want to grab an umbrella.",
            "Tomorrow looks even better though.",
        ],
    },
    {
        "name": "technical",
        "sentences": [
            "Okay, I found the issue.",
            "The server was returning a five hundred error because the database connection pool was exhausted.",
            "I have increased the pool size from ten to fifty and restarted the service.",
            "Everything should be back to normal now.",
        ],
    },
]


def tts_request(text):
    """Send text to TTS server, return (audio_bytes, ttfa_ms, total_ms)."""
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        "http://localhost:8100/tts",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttfa = None
    total_bytes = 0
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if ttfa is None and len(chunk) > 0:
                ttfa = (time.time() - t0) * 1000
            total_bytes += len(chunk)
    total_ms = (time.time() - t0) * 1000
    audio_s = (total_bytes / 4) / SAMPLE_RATE
    return audio_s, ttfa or total_ms, total_ms


print("=" * 80)
print("STRESS TEST: Simulating ivi conversation patterns")
print("=" * 80)

# Test 1: Sequential sentences (ivi pattern — each sentence as soon as LLM produces it)
print("\n--- Test 1: Sequential sentence-by-sentence (ivi pattern) ---")
for conv in conversations:
    print(f"\n  [{conv['name']}] ({len(conv['sentences'])} sentences)")
    conv_start = time.time()
    first_audio_at = None
    total_audio_s = 0
    total_gen_s = 0

    for i, sentence in enumerate(conv["sentences"]):
        audio_s, ttfa, total_ms = tts_request(sentence)
        total_audio_s += audio_s
        total_gen_s += total_ms / 1000
        if first_audio_at is None:
            first_audio_at = (time.time() - conv_start) * 1000 - total_ms + ttfa

        rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
        ahead = total_audio_s - total_gen_s
        status = f"ahead {ahead:.1f}s" if ahead > 0 else f"BEHIND {-ahead:.1f}s"

        print(
            f"    [{i}] TTFA={ttfa:>5.0f}ms total={total_ms:>6.0f}ms "
            f"audio={audio_s:>4.1f}s RTF={rtf:.3f} | {status} | \"{sentence[:40]}\""
        )

    conv_wall = (time.time() - conv_start) * 1000
    conv_rtf = (conv_wall / 1000) / total_audio_s if total_audio_s > 0 else 999
    print(
        f"    → Conv total: {conv_wall:.0f}ms wall, {total_audio_s:.1f}s audio, "
        f"RTF={conv_rtf:.3f}, first audio at {first_audio_at:.0f}ms"
    )

# Test 2: Rapid-fire (no pause between requests)
print("\n--- Test 2: Rapid-fire (10 requests, no pause) ---")
rapid_texts = [
    "Hey!",
    "Sure.",
    "Got it.",
    "What's on your mind?",
    "Yeah, that's pretty common.",
    "Let me check that for you.",
    "The weather looks great today.",
    "I'll update your calendar.",
    "Anything else?",
    "Sounds good!",
]

rtfs = []
ttfas = []
t_total = time.time()

for text in rapid_texts:
    audio_s, ttfa, total_ms = tts_request(text)
    rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
    rtfs.append(rtf)
    ttfas.append(ttfa)

total_wall = (time.time() - t_total) * 1000
total_audio = sum(
    tts_request(t)[0] for t in ["test"]
)  # dummy, we already have the data

print(f"  10 requests in {total_wall:.0f}ms")
print(f"  Avg RTF: {sum(rtfs)/len(rtfs):.3f}")
print(f"  Max RTF: {max(rtfs):.3f}")
print(f"  Avg TTFA: {sum(ttfas)/len(ttfas):.0f}ms")
print(f"  Min TTFA: {min(ttfas):.0f}ms")

# Test 3: Long text
print("\n--- Test 3: Long text (single request) ---")
long_text = (
    "I have been thinking about this for a while, and I think the best approach "
    "would be to start with a small prototype, test it with a few users, gather "
    "feedback, iterate on the design, and then gradually scale it up once we are "
    "confident that the core experience is solid."
)

audio_s, ttfa, total_ms = tts_request(long_text)
rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
print(f"  \"{long_text[:60]}...\"")
print(f"  TTFA={ttfa:.0f}ms, total={total_ms:.0f}ms, audio={audio_s:.1f}s, RTF={rtf:.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  All RTFs: avg={sum(rtfs)/len(rtfs):.3f}, max={max(rtfs):.3f}")
print(f"  All TTFAs: avg={sum(ttfas)/len(ttfas):.0f}ms")
print(f"  Server stable through {10 + sum(len(c['sentences']) for c in conversations) + 1} requests")
target = "✅ PASS" if sum(rtfs) / len(rtfs) <= 0.50 else "❌ FAIL"
print(f"  Target RTF ≤ 0.50: {target}")
