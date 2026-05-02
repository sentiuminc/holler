# Holler Voice Pack — 30 Voices

## Reference Texts

Canonical ref texts for voice cloning. Use one or both per voice — they're
complementary. The model uses the ref text + ref audio together to anchor voice
identity, so the text should match what's actually spoken in ref.wav.

### Reactive / Emotional
> "Oh wow, that actually worked! Can you believe it? I honestly thought we were going to have to start over but no, it just clicked."

**Currently used by:** Nora  
**What it captures:** Surprise, relief, natural excitement. Has rhythm variation,
a rising moment ("can you believe it?"), and a settling resolution ("it just clicked").
Good for voices that need emotional range and natural prosody.

### Assistant / Conversational
> "Okay, so I looked into it and here is what I found. The file you were working on got saved to your Downloads folder, not your Desktop. Want me to move it over, or would you rather keep it where it is?"

**Currently used by:** Kit, Dakota  
**What it captures:** Calm, helpful, measured cadence. Has a setup clause, an
information delivery beat, and a closing question. Good for voices that need to
sound natural in assistant-style speech — ivi-coded.

### Casual / Collaborative
> "Oh hey, that's actually a really good idea. Let me look into it — I think there might be a way to make this work that's way simpler than what we were thinking."

**Currently used by:** Joe  
**What it captures:** Casual enthusiasm, natural filler ("oh hey"), thinking-out-loud
quality. Has an em dash pause and a payoff clause. Good for warmer, more informal
voices — less polished than the assistant text, more human.

**Notes:**
- Texts are ~15-20 words — long enough to capture prosody, short enough for the cloner to stay consistent across 500 clips.
- Using multiple texts for one voice is fine — generate clips split across them. Variety in ref text helps the model generalize.
- Pick by character: reactive/warm → emotional text, calm/neutral → assistant text, casual/informal → collaborative text.

---

## Voice Roster

### US South & Appalachia
- **Dolly** — warm Tennessee honey, porch-swing storyteller
- **Hank** — deep Southern drawl, unhurried, says "y'all" unironically
- **Jolene** — bright, fast-talking Southern belle
- **Buck** — gravel-voiced Florida man energy, been through some stuff

### US Northeast
- **Vinnie** — fast Jersey/Brooklyn, talks with his hands
- **Nora** — clear, deep female voice, direct and natural

### US West
- **Sage** — mellow California, surfer-adjacent but smart
- **Dakota** — Colorado/Pacific Northwest outdoorsy, steady
- **Bryce** — LA tech bro cadence, uptalks everything

### US Midwest
- **Clyde** — Minnesota nice, warm dad energy, "oh ya you betcha"
- **Ruthie** — cheerful Chicago, a little nasal, a lot of heart

### Texas
- **Travis** — big Texas baritone, confident, takes his time
- **Bonnie** — quick Austin wit, modern Texan, not a stereotype

### British RP / London
- **Oliver** — crisp received pronunciation, BBC newsreader energy
- **Pippa** — posh but playful, thinks everything is "brilliant"
- **Alfie** — cheeky East London, quick, a bit of swagger

### British Regional
- **Angus** — Scottish, deep, rolling R's, sounds like he'd survive a blizzard
- **Rhys** — Welsh, musical lilt, warm
- **Gemma** — Manchester, northern warmth, straight shooter

### Australian
- **Callum** — laid-back Sydney bloke, everything's "no worries"
- **Tessa** — sharp Melbourne, dry humor, fast

### New Zealand
- **Aroha** — Kiwi accent, gentle, grounded

### South African
- **Stellan** — Cape Town English, distinctive vowels, worldly

### Canadian
- **Maple** — friendly Toronto, clean neutral-ish
- **Fraser** — Albertan, outdoorsy, slight Canadian raise

### Wild Cards
- **Kit** — androgynous, could be anyone, clean mid-Atlantic
- **Wren** — whispery, ASMR-adjacent, intimate
- **Knox** — deep authoritative narrator voice, audiobook mode
- **Pearl** — elderly Southern grandmother, wise, slow
- **Ziggy** — high-energy, young, podcast host energy

---

## Slot Assignments

- Katie — slot 3000 (Cartesia-sourced, trained, production-ready)
- Joe — slot 3001 (VoiceDesign-sourced, trained, quality issues in multi-voice)
- Joseph — slot 3002 (VoiceDesign-sourced, designed, not yet trained) — deep bass, authoritative patriarch

The remaining 27 voices will be assigned slots 3003-3029 during training. Katie may be renamed to one of these if she fits a character.

## Design Process

Use `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` to generate reference audio for each voice from a character description. Multiple rounds of candidates per voice, pick the best. Clone 500 training clips per voice via 1.7B-Base-bf16 on GPU (Vast.ai). Enhance with `enhance_clean.py`. Manually curate with clip tinder before training.
