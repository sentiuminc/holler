# Holler Voice Pack — 30 Voices

## US South & Appalachia
- **Dolly** — warm Tennessee honey, porch-swing storyteller
- **Hank** — deep Southern drawl, unhurried, says "y'all" unironically
- **Jolene** — bright, fast-talking Southern belle
- **Buck** — gravel-voiced Florida man energy, been through some stuff

## US Northeast
- **Vinnie** — fast Jersey/Brooklyn, talks with his hands
- **Nora** — sharp Boston, no-nonsense, drops her R's

## US West
- **Sage** — mellow California, surfer-adjacent but smart
- **Dakota** — Colorado/Pacific Northwest outdoorsy, steady
- **Bryce** — LA tech bro cadence, uptalks everything

## US Midwest
- **Clyde** — Minnesota nice, warm dad energy, "oh ya you betcha"
- **Ruthie** — cheerful Chicago, a little nasal, a lot of heart

## Texas
- **Travis** — big Texas baritone, confident, takes his time
- **Bonnie** — quick Austin wit, modern Texan, not a stereotype

## British RP / London
- **Oliver** — crisp received pronunciation, BBC newsreader energy
- **Pippa** — posh but playful, thinks everything is "brilliant"
- **Alfie** — cheeky East London, quick, a bit of swagger

## British Regional
- **Angus** — Scottish, deep, rolling R's, sounds like he'd survive a blizzard
- **Rhys** — Welsh, musical lilt, warm
- **Gemma** — Manchester, northern warmth, straight shooter

## Australian
- **Callum** — laid-back Sydney bloke, everything's "no worries"
- **Tessa** — sharp Melbourne, dry humor, fast

## New Zealand
- **Aroha** — Kiwi accent, gentle, grounded

## South African
- **Stellan** — Cape Town English, distinctive vowels, worldly

## Canadian
- **Maple** — friendly Toronto, clean neutral-ish
- **Fraser** — Albertan, outdoorsy, slight Canadian raise

## Wild Cards
- **Kit** — androgynous, could be anyone, clean mid-Atlantic
- **Wren** — whispery, ASMR-adjacent, intimate
- **Knox** — deep authoritative narrator voice, audiobook mode
- **Pearl** — elderly Southern grandmother, wise, slow
- **Ziggy** — high-energy, young, podcast host energy

---

**Slot assignments:**
- Katie — slot 3000 (Cartesia-sourced, trained, production-ready)
- Joe — slot 3001 (VoiceDesign-sourced, trained, quality issues in multi-voice)
- Joseph — slot 3002 (VoiceDesign-sourced, designed, not yet trained) — deep bass, authoritative patriarch

The remaining 27 voices will be assigned slots 3003-3029 during training. Katie may be renamed to one of these if she fits a character.

**Design process:** Use `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` to generate reference audio for each voice from a character description. Multiple rounds of candidates per voice, pick the best. Clone 385 training clips per voice via 1.7B-Base-8bit locally on Mac.
