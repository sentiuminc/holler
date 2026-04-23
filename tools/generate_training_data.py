"""Generate training data for Qwen3-TTS 0.6B fine-tuning using 1.7B Base voice cloning."""
import time
import os
import json
import numpy as np
import soundfile as sf
import mlx.core as mx

REF_AUDIO = "/Users/nagy/Downloads/ivi-tts-benchmark/clone-1.7b/ref_v2_24k.wav"
REF_TEXT = "Hey, so I've been thinking about this for a while now. The thing is, you don't really notice how much it matters until you actually try it yourself. It's one of those subtle differences that just clicks."
OUTPUT_DIR = os.path.expanduser("~/Downloads/ivi-tts-training-data")
os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)

# Copy ref audio into the training data directory
import shutil
ref_dest = os.path.join(OUTPUT_DIR, "ref.wav")
shutil.copy2(REF_AUDIO, ref_dest)

# Diverse training texts — varied lengths, structures, emotions, topics
# Goal: 200+ clips covering natural conversational speech patterns
TEXTS = [
    # Short responses (2-4 words)
    "Got it.",
    "Sure thing.",
    "Absolutely.",
    "Not quite.",
    "Let me check.",
    "One moment.",
    "That works.",
    "Good idea.",
    "Interesting.",
    "I see.",

    # Short sentences (5-10 words)
    "I'll take care of that right now.",
    "That's exactly what I was thinking.",
    "Let me pull that up for you.",
    "Here's what I found so far.",
    "Would you like me to continue?",
    "I think we should try a different approach.",
    "That doesn't look quite right to me.",
    "Everything is running smoothly on my end.",
    "I'm ready whenever you are.",
    "Let me know if you need anything else.",
    "Sure, I can help with that.",
    "Give me just a second.",
    "That's a great question actually.",
    "I hadn't thought of it that way.",
    "You make a really good point.",

    # Medium sentences (10-20 words)
    "I've been looking into this and I think there might be a simpler way to handle it.",
    "The main issue seems to be with how the data is being processed on the backend.",
    "I ran the tests and everything passed, but I want to double check one more thing.",
    "Based on what you've told me, I think the best option would be to start fresh.",
    "It's actually not as complicated as it looks once you break it down step by step.",
    "I noticed something interesting while reviewing the logs from earlier today.",
    "The good news is that we already have most of what we need to make this work.",
    "I just finished setting everything up and it should be ready to go now.",
    "Let me walk you through what I did so you can see the full picture.",
    "I want to make sure I understand correctly before making any changes.",
    "That's one of those things that seems simple but has a lot of edge cases.",
    "I've seen this pattern before and I know exactly how to fix it.",
    "The performance numbers are looking really good compared to what we had before.",
    "I think the key insight here is that we don't need to handle every case.",
    "Would it help if I put together a quick summary of where things stand?",
    "I'm going to need a bit more context before I can give you a solid answer.",
    "Everything looks good on my end, but let me verify one more time to be sure.",
    "There are a few different ways we could approach this, depending on the priority.",
    "I just realized there might be a conflict with what we set up earlier.",
    "The data is all there, we just need to figure out the right way to display it.",

    # Longer explanations (20-35 words)
    "So the way this works is pretty straightforward. You give it the input, it processes everything in the background, and then you get the results back in real time.",
    "I've been thinking about this problem from a different angle and I think the real issue isn't the code itself, it's how we're structuring the data flow.",
    "The reason it's taking longer than expected is because each request needs to go through multiple validation steps before it can be processed by the main system.",
    "What I'd suggest is that we start with the simplest possible version, get that working reliably, and then gradually add the more complex features on top.",
    "From what I can tell, the bottleneck is happening right at the point where the system needs to decide which handler to route the request to.",
    "I looked into this earlier and the documentation is a bit misleading. The actual behavior is slightly different from what they describe on the website.",
    "The tricky part about this is that it works perfectly fine in testing but behaves differently in production because of how the environment is configured.",
    "I want to be upfront about this. There's a trade off here between speed and accuracy, and we need to decide which one matters more for this specific use case.",
    "One thing I noticed is that the error messages aren't very helpful right now. If we improve those, debugging would be so much easier going forward.",
    "The way I see it, we have two main options. We can either fix the existing system or build something new from scratch. Each has its own set of tradeoffs.",

    # Questions
    "What would you like me to focus on first?",
    "Should I go ahead and make those changes now?",
    "Do you want the short version or the full breakdown?",
    "Have you tried restarting the application?",
    "What's the deadline we're working with here?",
    "Is there anything else you'd like me to look into?",
    "How does that sound to you?",
    "Want me to run through it one more time?",
    "Does this match what you were expecting?",
    "Should we tackle this now or come back to it later?",
    "Can you tell me a bit more about what you're trying to do?",
    "What's the most important thing to get right here?",
    "Are you happy with how this turned out?",
    "Would you prefer a more detailed explanation?",
    "Is there a specific part that's not working the way you expected?",

    # Emotional variety — enthusiastic
    "Oh, that's awesome! I didn't expect it to work this well on the first try.",
    "This is really exciting. I think we're onto something here.",
    "I love that idea. Let's absolutely do that.",
    "Wow, the results are way better than I thought they'd be!",
    "Yes! That's exactly the kind of thing I was hoping for.",

    # Emotional variety — thoughtful/careful
    "Hmm, that's an interesting edge case. Let me think about how to handle it properly.",
    "I want to be careful here because any changes we make could affect other parts of the system.",
    "Let me take a step back and think about whether this is really the right approach.",
    "I'm not entirely sure about this one. There are some risks we should consider first.",
    "This requires a bit of nuance. The answer isn't as straightforward as it might seem.",

    # Emotional variety — empathetic
    "I completely understand your frustration. Let's see what we can do to fix this.",
    "No worries at all, that's a totally reasonable question.",
    "I hear you. That does sound like a really annoying issue to deal with.",
    "Take your time, there's no rush. I'll be here whenever you're ready.",
    "I get it. Sometimes these things just don't work the way they should.",

    # Emotional variety — matter-of-fact
    "The build succeeded. No errors or warnings.",
    "Total cost for the month was twelve dollars and forty seven cents.",
    "The server has been running for seventy two hours without any issues.",
    "Version three point five was released on Tuesday.",
    "Memory usage is currently at sixty eight percent.",

    # Technical content
    "The API endpoint accepts both GET and POST requests, but you'll get better performance with POST for larger payloads.",
    "I set up the database migration to run automatically during deployment. It handles both the forward and rollback cases.",
    "The latency numbers are looking great. We're averaging about forty five milliseconds per request under normal load.",
    "The cache hit rate jumped to ninety two percent after we switched to the new eviction policy.",
    "I configured the load balancer to distribute traffic evenly across all three instances.",
    "The authentication flow uses OAuth two with refresh tokens that expire after thirty days.",
    "We're using a combination of unit tests and integration tests to cover the main scenarios.",
    "The webhook fires every time a new event comes in, and we process them in order using a queue.",
    "I added rate limiting at fifty requests per minute per user to prevent abuse.",
    "The backup runs every six hours and keeps the last seven days of snapshots.",

    # Conversational filler / natural speech
    "So yeah, that's basically the gist of it.",
    "Anyway, where were we?",
    "Right, so as I was saying.",
    "Oh wait, I just thought of something.",
    "Actually, hold on. Let me reconsider that.",
    "You know what, I think you're right about that.",
    "Okay so here's the thing.",
    "Alright, let me think about this for a moment.",
    "Fair enough, I can see where you're coming from.",
    "Yeah, that makes total sense when you put it that way.",

    # Instructions / directing
    "First, open the settings page and look for the advanced options section.",
    "Go ahead and click the blue button in the top right corner.",
    "Try refreshing the page and see if that fixes the issue.",
    "You'll want to save your work before we make any changes.",
    "Copy the token from that page and paste it into the configuration file.",
    "Navigate to the dashboard and select the second tab from the left.",
    "Scroll down until you see the section labeled connection settings.",
    "Make sure you have the latest version installed before we continue.",
    "Click on your profile icon, then go to account settings.",
    "Double check that the file name matches exactly, including the extension.",

    # Storytelling / narrative
    "So what happened was, the system was running fine all morning, and then around two o'clock everything just stopped.",
    "I remember when we first started working on this. Nobody thought it would actually turn into what it is today.",
    "The funny thing is, the solution was right there the whole time. We just weren't looking at it from the right angle.",
    "It took us about three weeks to figure out what was going on. Turns out it was a single misconfigured setting.",
    "We went through maybe five or six different approaches before finding one that actually worked consistently.",

    # Acknowledgments / transitions
    "Perfect, that's all done.",
    "Great, moving on to the next step.",
    "Alright, that should take care of it.",
    "Okay, we're all set on that front.",
    "Done. What would you like to tackle next?",
    "Good to go. Everything is in place.",
    "That wraps up the first part. Ready for the next?",
    "All finished. Let me know how it looks on your end.",
    "Cool, that's one less thing to worry about.",
    "Nice, we're making good progress here.",

    # Numbers and specifics
    "The file is about fourteen megabytes, so it should download pretty quickly.",
    "We have roughly two hundred and fifty active users at any given time.",
    "The response time improved from three hundred milliseconds down to about seventy five.",
    "I'd say we're about eighty percent done with the main features.",
    "There are seventeen open issues in the backlog, but only four of them are critical.",
    "The meeting is scheduled for three thirty tomorrow afternoon.",
    "We've processed over ten thousand requests since the last deployment.",
    "The new server has sixty four gigabytes of RAM and sixteen cores.",
    "Battery is at forty two percent. You might want to plug in soon.",
    "It's currently eleven degrees outside with a chance of rain later tonight.",

    # Varied sentence starters
    "Honestly, I think this is the best approach we've come up with so far.",
    "Between you and me, I think we might be overcomplicating this.",
    "At the end of the day, what matters most is that it works reliably.",
    "If I'm being honest, I wasn't sure this would work at first.",
    "Looking at the bigger picture, I think we're in really good shape.",
    "Speaking from experience, these kinds of issues usually have a simple fix.",
    "In my opinion, the current setup is already pretty solid.",
    "Just to be clear, this won't affect anything that's already running.",
    "For what it's worth, I think your instinct on this was correct.",
    "Long story short, we need to update the configuration and redeploy.",

    # Multi-sentence (natural pauses)
    "Okay, so I looked into the issue. It turns out the problem was on our side, not theirs. Easy fix once you know where to look.",
    "Here's my suggestion. Let's keep the current version running for now and work on the update in parallel. That way we don't risk any downtime.",
    "I just checked the logs. Everything looks clean. No errors, no warnings, no timeout issues. Whatever you did seems to have fixed it.",
    "So there's good news and not so good news. The good news is the core functionality works perfectly. The not so good news is we need to redo the interface.",
    "I ran the benchmark three times to make sure. The results were consistent each time. We're getting about twice the throughput compared to the old system.",
]

print(f"Total texts: {len(TEXTS)}")
print(f"Loading Qwen3-TTS 1.7B Base (8-bit)...")
t0 = time.time()
from mlx_audio.tts import load
model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
print(f"Model loaded in {time.time()-t0:.1f}s")

print("Warmup pass...")
for _ in model.generate(text="Hello.", ref_audio=REF_AUDIO, ref_text=REF_TEXT, language="en", temperature=0.6):
    pass
print("Warmup done.\n")

# 1 second of silence at 24kHz for appending to each clip
silence_1s = np.zeros(24000, dtype=np.float32)

manifest = []
total_audio_duration = 0
failed = []

for i, text in enumerate(TEXTS):
    label = f"clip_{i+1:04d}"
    print(f"[{i+1}/{len(TEXTS)}] {text[:60]}...")

    t0 = time.time()
    chunks = []

    try:
        for result in model.generate(
            text=text,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            language="en",
            temperature=0.6,
        ):
            audio = np.array(result.audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            chunks.append(audio)

        full_audio = np.concatenate(chunks)

        # Append 1s silence for clean EOS training
        full_audio = np.concatenate([full_audio, silence_1s])

        duration = len(full_audio) / 24000
        total_ms = (time.time() - t0) * 1000

        out_path = os.path.join(OUTPUT_DIR, "audio", f"{label}.wav")
        sf.write(out_path, full_audio, 24000)

        manifest.append({
            "audio": f"./audio/{label}.wav",
            "text": text,
            "ref_audio": "./ref.wav",
        })
        total_audio_duration += duration
        print(f"  {total_ms:.0f}ms | {duration:.1f}s | Total: {total_audio_duration/60:.1f}min")

    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append((i, text, str(e)))

# Write JSONL manifest
jsonl_path = os.path.join(OUTPUT_DIR, "train.jsonl")
with open(jsonl_path, "w") as f:
    for entry in manifest:
        f.write(json.dumps(entry) + "\n")

print(f"\n{'='*60}")
print(f"Done! Generated {len(manifest)} clips")
print(f"Total audio: {total_audio_duration/60:.1f} minutes")
print(f"Failed: {len(failed)}")
print(f"Output: {OUTPUT_DIR}/")
print(f"Manifest: {jsonl_path}")
if failed:
    print(f"\nFailed clips:")
    for idx, txt, err in failed:
        print(f"  [{idx}] {txt[:50]}... — {err}")
