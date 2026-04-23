# Session Log: 2026-04-23 — Voice Exploration & Joseph Design

**Project:** Holler
**Session ID:** `458be8c7-aacc-4d6c-9176-df61013bc663`
**What happened:** Explored voice design, picked Joseph as voice #3, experimented with Russian accents via VoiceDesign and voice cloning.

---

## What I Did

- Played all Joe v7 PyTorch + MLX samples for Chris to evaluate
- Generated a fresh Joe sample with a long MLX profiling text
- Designed 4 "Russian mafia" voice candidates using VoiceDesign (boris01-04), varying from mob boss to smooth oligarch to enforcer to retired KGB
- Chris picked boris04 (deep bass, old-school patriarch) — saved as **Joseph**, voice #3, slot 3002
- Generated 8 additional boris04 samples across varied texts to confirm the voice holds up
- Cloned boris04 onto ivi assistant sentences (weather, Uber, calendar) using 1.7B-Base — saved as Joseph's cloned samples
- Also cloned sample03 and sample05 from the boris04 batch — saved as `_unsorted/random-male-01` and `random-male-02` (not chosen, not discarded)
- Attempted Russian accent in English via VoiceDesign: tried `language='russian'` with English text, tried heavily Russian-loaded instructs. Result: shifted timbre toward "Russian czar" but didn't achieve actual accent. VoiceDesign controls timbre/prosody, not phoneme-level accent.
- Cloned a Galkin comedy sketch (YouTube) — Russian kid speaking English with heavy accent. Cloning worked but discovered **reference bleed**: model echoes the tail of the reference audio into generated speech. Fix: trim ref to end cleanly. Saved as `_unsorted/russian-kid-01` with notes.

## What I Learned

- VoiceDesign's instruct controls voice character (pitch, register, warmth, pace) but cannot inject accents. Accent requires actual accented reference audio for cloning.
- Reference bleed is a real gotcha with voice cloning — the last few words of the ref clip leak into the start of generated speech. Trimming the ref audio resolves it.
- `language='russian'` with English text slightly shifts pronunciation but doesn't produce a proper Russian accent.
- Chris likes deep, authoritative, weighty voices. Joseph was an instant pick.

## For Next Time

- Joseph needs training data generated (385 clips via 1.7B-Base cloning, same as Katie/Joe)
- Then single-voice SFT at lr=1e-7, 2 epochs
- Russian accent voices would need real accented English reference audio (movie clips, interviews) — VoiceDesign alone can't do it
- The `_unsorted/` pattern works well for parking interesting voices that aren't committed to the 30-voice plan

## Files Changed

- `voices/joseph/` — new voice directory (ref.wav + 12 candidates)
- `voices/_unsorted/random-male-01/` — unchosen male voice + 8 cloned samples
- `voices/_unsorted/random-male-02/` — unchosen male voice + 8 cloned samples
- `voices/_unsorted/russian-kid-01/` — Galkin clone experiment + notes
- `voices/VOICES.md` — added Joseph slot assignment, updated slot table
- `CLAUDE.md` — updated voice count to 3/30, added Joseph to structure and key details
