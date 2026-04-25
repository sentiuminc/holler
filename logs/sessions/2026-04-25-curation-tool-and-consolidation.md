# Session Log: 2026-04-25 — Curation Tool + Consolidation Decision

**Project:** Holler
**Session ID:** `f22bdb1d-7165-4c8f-941d-f0b8df8cb5cb`
**What happened:** Built the audio curation tool, Chris curated ~250/385 Katie clips, discovered training data lacks variety, decided to move holler back into ivi.

---

## What I Did

### Curation Tool ("Clip Tinder")

Built `tools/curate_clips.py` — a browser-based tool at localhost:8200 for swiping through training clips. Controls: ← reject, → keep, ↓ maybe, Shift undo. Auto-plays audio, persists every decision to `curation.json`, resumes across sessions. Three modes: Undecided (default), Maybes (review round), All (browse everything). Export button generates `train_curated.jsonl` with only keeps. Pure Python stdlib, no dependencies.

Chris used it and got through ~250 of 385 Katie clips. ~5% reject rate, ~21% maybe rate.

### Training Data Variety Analysis

Chris noticed after listening to 250 clips that Katie's training data sounds monotonous. I analyzed the text distribution: 94% end with periods, 6% questions, **1 exclamation mark** in 385 clips. Almost all sentences are calm explanatory tone. Plan was to generate 50 new clips with emotionally varied text (no periods, questions/exclamations only) at temperature 0.8 — but paused because of the consolidation decision.

### Consolidation: Holler → Back into ivi

Chris decided holler should move back into ivi. The standalone repo was premature — the venv was split (holler had enhancement deps but no mlx-audio, ivi had mlx-audio but no enhancement deps). Maintaining two environments was pointless friction.

Updated `requirements.txt` in both locations with the full dependency list: mlx-audio, torch 2.6 (pinned for DeepFilterNet), torchaudio 2.6, clearvoice, deepfilternet, noisereduce, scipy, soundfile, numpy.

### Cleanup

Deleted dead-end experiments: `speech-swift/` (rejected Swift TTS integration), `tools/ap-bwe/` (couldn't install — manual weight download), `swift-test/` (depended on speech-swift), `inference/test_trimming.py` (referenced ivi sidecar code).

## What I Learned

- Chris listens to training data carefully and catches quality issues that metrics can't. The variety problem (94% periods) was invisible until someone actually sat with 250 clips.
- Separating repos too early creates real friction — split venvs, duplicated requirements, confusion about which directory to work from.
- Chris prefers flat files for listening (confirmed again — the curation tool puts everything at one level).

## For Next Time

- **All work now happens from `~/Desktop/Files/AI/ivi/holler/`**, not this repo. This repo is archived.
- The curation tool and curation.json need to be carried over to ivi.
- After the move: consolidate into one venv with all deps (mlx-audio + enhancement stack).
- Generate 50+ varied training clips (emotionally loaded, no periods, temp 0.8) for Katie.
- Design new female voices (Jolene, Bonnie from VOICES.md were mentioned).
- Finish Katie curation (~135 clips remaining), then curate Joe.
- Retrain with curated + enhanced + varied data.

## Files Changed

- `tools/curate_clips.py`: New — audio curation tool
- `requirements.txt`: Updated with full dependency list
- `docs/arena-submission.md`: New — arena submission notes
- `voices/katie/training-data/curation.json`: New — curation progress
- Deleted: `speech-swift/`, `tools/ap-bwe/`, `swift-test/`, `inference/test_trimming.py`
- Memory: `project_origin.md` updated (holler moving back into ivi)
