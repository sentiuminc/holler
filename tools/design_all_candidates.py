#!/usr/bin/env python3
"""Design candidates for ALL remaining voices from VOICES.md.

Voice assistant framing — these are Siri-like voices with regional personality.
Each voice gets 8 instruct prompts. All speak the same assistant-style target text.

Output: ~/Downloads/voice-candidates/ with all .wav files + index.txt
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Okay, so I looked into it and here's what I found. The file you were working on "
    "got saved to your Downloads folder, not your Desktop. Want me to move it over, "
    "or would you rather keep it where it is?"
)

OUT_DIR = Path("/Users/nagy/Downloads/voice-candidates")

# ============================================================================
# US NORTHEAST
# ============================================================================

VINNIE = [
    ("vinnie01_fast_jersey",
     "Fast-talking male voice assistant with a Jersey accent. Early thirties. Mid-range "
     "baritone, punchy delivery. Quick cadence, clipped vowels, strong consonants. "
     "Helpful and direct, no fluff. Sounds like he's got places to be but will help you first."),

    ("vinnie02_brooklyn_warm",
     "Male voice assistant with a warm Brooklyn accent. Early thirties. Medium register, "
     "friendly and fast. Distinct New York vowels, talks with energy. Helpful, a little "
     "impatient but in a charming way. Clear despite the speed."),

    ("vinnie03_sharp_ny",
     "Sharp, quick male voice assistant with a New York accent. Late twenties. Higher "
     "baritone, bright and punchy. Fast tempo, crisp delivery. Sounds street-smart and "
     "competent. Jersey-Brooklyn energy without being aggressive."),

    ("vinnie04_confident_east",
     "Confident male voice assistant from the East Coast. Early thirties. Mid-range, "
     "fast-talking, New York inflection. Strong consonants, quick vowels. He sounds like "
     "he already knows the answer before you finish asking. Helpful and efficient."),

    ("vinnie05_smooth_jersey",
     "Smooth-talking male voice assistant with a light Jersey accent. Late twenties. "
     "Medium register, quick but controlled delivery. Charming, confident, a little "
     "swagger in the tone. Clear articulation despite the speed."),

    ("vinnie06_street_smart",
     "Street-smart male voice assistant, early thirties. Mid-range baritone with New York "
     "energy. Quick cadence, direct delivery. Not rough, just fast and real. Brooklyn "
     "vowels, helpful attitude. Sounds like your smartest friend from the neighborhood."),

    ("vinnie07_energetic_ny",
     "Energetic male voice assistant with a New York accent. Early thirties. Bright "
     "baritone, fast and friendly. Talks like every sentence matters. Strong East Coast "
     "inflection, crisp consonants. Helpful with real personality."),

    ("vinnie08_quick_charm",
     "Quick, charming male voice assistant from Jersey. Late twenties. Mid-high register, "
     "rapid delivery with natural New York rhythm. Not aggressive, just naturally fast. "
     "Warm underneath the speed. An assistant who gets things done quickly."),
]

# ============================================================================
# US WEST
# ============================================================================

SAGE = [
    ("sage01_mellow_cali",
     "Mellow male voice assistant with a California vibe. Late twenties. Mid-range, "
     "smooth and relaxed. Easy pace, warm tone. Sounds laid-back but smart. "
     "Clear, natural delivery with no affectation."),

    ("sage02_surfer_smart",
     "Relaxed male voice assistant, late twenties. Smooth mid-range, California-mellow "
     "delivery. Unhurried but articulate. Warm, friendly, sounds like he thinks before "
     "he speaks. Natural, unforced ease."),

    ("sage03_pacific_calm",
     "Calm male voice assistant with a Pacific coast sensibility. Early thirties. "
     "Medium register, smooth and steady. Relaxed pacing, clear diction. Sounds "
     "grounded and thoughtful. No rush, no filler."),

    ("sage04_warm_west",
     "Warm, easy-going male voice assistant. Late twenties. Mid-range, smooth timbre. "
     "California-relaxed cadence, clear and helpful. Sounds like sunshine in voice form. "
     "Friendly without trying too hard."),

    ("sage05_chill_clear",
     "Chill but clear male voice assistant, early thirties. Medium register, natural "
     "delivery. Laid-back West Coast feel but never lazy. Every word lands clearly. "
     "Warm, approachable, quietly confident."),

    ("sage06_deep_mellow",
     "Deep, mellow male voice assistant, late twenties. Lower mid-range, smooth and "
     "warm. California ease in every syllable. Relaxed pace, rich tone. Sounds like "
     "a thoughtful friend who happens to know everything."),

    ("sage07_golden_state",
     "Smooth male voice assistant with a West Coast character. Early thirties. Medium "
     "register, warm and even. Natural, relaxed delivery. Clear, helpful, with a "
     "calm confidence. Not performative, just naturally easy."),

    ("sage08_ocean_calm",
     "Ocean-calm male voice assistant, late twenties. Smooth mid-range, very relaxed "
     "pacing. Warm, clear, grounded. Sounds like someone who meditates but also reads "
     "a lot. California vibe without the cliche."),
]

DAKOTA = [
    ("dakota01_steady_outdoor",
     "Steady male voice assistant with an outdoor sensibility. Early thirties. Clean "
     "mid-range, even delivery. Sounds grounded and capable. Pacific Northwest neutral, "
     "clear articulation. Dependable and calm."),

    ("dakota02_mountain_clear",
     "Clear, steady male voice assistant. Early thirties. Medium register, clean and "
     "natural. Colorado-neutral accent, no frills. Speaks with quiet confidence. "
     "An assistant who sounds like he could also build a cabin."),

    ("dakota03_trail_guide",
     "Grounded male voice assistant, early thirties. Mid-range baritone, steady and "
     "clear. Neutral Western accent, unhurried but efficient. Warm, competent, no "
     "nonsense. Sounds reliable in any situation."),

    ("dakota04_pacific_nw",
     "Pacific Northwest male voice assistant, early thirties. Clean mid-range, "
     "natural delivery. Not too fast, not too slow. Clear, helpful, genuine. "
     "Sounds like someone who values simplicity and getting things right."),

    ("dakota05_colorado_steady",
     "Steady, capable male voice assistant. Early thirties. Medium baritone, clean "
     "and even. Neutral American with a Western groundedness. Patient, clear, "
     "helpful. Sounds like the most competent person in any room."),

    ("dakota06_calm_west",
     "Calm Western male voice assistant, early thirties. Mid-range, clean timbre. "
     "Steady pace, natural delivery. Sounds outdoorsy and intelligent. Clear "
     "articulation, warm but not soft. A steady hand."),

    ("dakota07_alpine_clear",
     "Clear, clean male voice assistant, early thirties. Medium register, bright "
     "but not flashy. Neutral accent, efficient delivery. Sounds fresh and "
     "capable. No frills, just helpful clarity."),

    ("dakota08_evergreen",
     "Evergreen male voice assistant, early thirties. Warm mid-range, natural and "
     "grounded. Pacific Northwest ease, clear diction. Dependable, steady, "
     "genuinely helpful. A voice you'd want on a long trip."),
]

BRYCE = [
    ("bryce01_tech_bro",
     "Male voice assistant with LA tech energy. Late twenties. Higher baritone, "
     "fast, enthusiastic delivery. Slight upward inflection. Sounds excited about "
     "everything but genuinely helpful. Clear, modern, quick."),

    ("bryce02_startup_energy",
     "Energetic male voice assistant, late twenties. Bright mid-range, quick cadence. "
     "LA-modern delivery with slight uptalk. Sounds like he just discovered something "
     "amazing. Friendly, fast, a little breathless from enthusiasm."),

    ("bryce03_podcast_host",
     "Male voice assistant with podcast host energy. Late twenties. Mid-range, "
     "articulate, enthusiastic. Quick delivery, engaging cadence. Slight California "
     "inflection. Makes everything sound interesting. Clear and bright."),

    ("bryce04_upbeat_la",
     "Upbeat male voice assistant from LA, late twenties. Higher register, bright "
     "and fast. Natural uptalk on some phrases. Sounds genuinely stoked to help. "
     "Modern, young, energetic. Clear despite the speed."),

    ("bryce05_silicon_smooth",
     "Smooth, fast male voice assistant. Late twenties. Mid-range, polished delivery "
     "with California flair. Quick cadence, confident tone. Sounds like he reads "
     "TechCrunch but is actually helpful. Modern and clean."),

    ("bryce06_venice_bright",
     "Bright male voice assistant, late twenties. Higher mid-range, quick and clear. "
     "LA energy, slight uptalk. Enthusiastic without being annoying. Sounds young, "
     "smart, and genuinely interested in helping."),

    ("bryce07_optimist",
     "Optimistic male voice assistant, late twenties. Bright baritone, fast delivery. "
     "California-positive energy. Everything sounds like good news. Upward inflection "
     "on statements, clear articulation. Fun and helpful."),

    ("bryce08_hustle_charm",
     "Charming, fast male voice assistant with LA energy. Late twenties. Mid-range, "
     "quick and smooth. Modern cadence with slight uptalk. Sounds like he knows "
     "everyone and wants to help you too. Bright, clear, engaging."),
]

# ============================================================================
# US MIDWEST
# ============================================================================

CLYDE = [
    ("clyde01_minnesota_nice",
     "Warm, friendly male voice assistant with a Minnesota accent. Mid-fifties. "
     "Mid-range, rounded vowels, warm delivery. Speaks with genuine kindness. "
     "Upper Midwest inflection, unhurried pace. A warm, fatherly helper."),

    ("clyde02_dad_energy",
     "Male voice assistant with warm dad energy. Mid-fifties. Medium register, "
     "gentle and steady. Light Minnesota accent, rounded 'O' sounds. Patient, "
     "helpful, never condescending. Sounds like he'd also make you breakfast."),

    ("clyde03_heartland_warm",
     "Warm heartland male voice assistant, early fifties. Mid-range, friendly and "
     "steady. Minnesota nice in every syllable. Clear, rounded delivery. "
     "Genuinely helpful, patient, makes you feel at home."),

    ("clyde04_lake_calm",
     "Calm, warm male voice assistant, mid-fifties. Medium register, smooth with "
     "a light Upper Midwest accent. Unhurried, gentle delivery. Sounds like a "
     "man sitting by a lake who has all the time for you."),

    ("clyde05_gentle_north",
     "Gentle male voice assistant from the Upper Midwest. Early fifties. Mid-range, "
     "warm and rounded. Light Minnesota vowels, easy pace. Sounds genuinely "
     "interested in helping. Kind without being soft."),

    ("clyde06_reliable_warm",
     "Reliable, warm male voice assistant, mid-fifties. Medium baritone, steady "
     "and comforting. Midwestern inflection, clear delivery. Sounds dependable and "
     "friendly. The kind of voice that makes problems feel smaller."),

    ("clyde07_cheerful_midwest",
     "Cheerful male voice assistant with a Midwest accent. Early fifties. Mid-range, "
     "bright and warm. Minnesota-friendly cadence, rounded vowels. Upbeat but not "
     "hyperactive. Genuinely pleasant to listen to."),

    ("clyde08_pops_helper",
     "Warm, paternal male voice assistant, mid-fifties. Medium register, gentle "
     "delivery. Light Upper Midwest accent, clear and unhurried. Sounds like the "
     "nicest man you've ever met. Helpful, patient, full of warmth."),
]

RUTHIE = [
    ("ruthie01_chicago_heart",
     "Female voice assistant with a warm Chicago accent. Late thirties. Medium register, "
     "bright and friendly. Slightly nasal in a charming way. Quick, helpful cadence. "
     "Sounds like she genuinely cares about getting it right for you."),

    ("ruthie02_midwest_bright",
     "Bright, cheerful female voice assistant from Chicago. Late thirties. Higher "
     "mid-range, clear and warm. Light Midwest accent, friendly delivery. Enthusiastic "
     "about helping. A touch of nasal warmth. Genuine and likeable."),

    ("ruthie03_windy_city",
     "Female voice assistant with Windy City warmth. Early forties. Medium register, "
     "direct and friendly. Chicago accent adds character. Quick but clear delivery. "
     "Sounds practical and kind. Straight-talking helper."),

    ("ruthie04_heartfelt_chi",
     "Heartfelt female voice assistant, late thirties. Mid-range, warm and genuine. "
     "Light Chicago inflection, clear delivery. She sounds like she means every word. "
     "Helpful, bright, a little spirited."),

    ("ruthie05_neighborhood",
     "Neighborhood-friendly female voice assistant, early forties. Medium register, "
     "warm and slightly nasal. Chicago cadence, direct delivery. Sounds like she'd "
     "bring you soup and also fix your computer. Real and helpful."),

    ("ruthie06_sharp_warm",
     "Sharp but warm female voice assistant from Chicago. Late thirties. Mid-high "
     "register, clear and quick. Midwest directness with genuine warmth. Slightly "
     "nasal edge adds character. Efficient and friendly."),

    ("ruthie07_north_side",
     "Female voice assistant with a North Side Chicago vibe. Late thirties. Bright, "
     "clear, medium register. Quick helpful delivery. Light Midwest accent. Sounds "
     "smart, friendly, and no-nonsense. Gets to the point with a smile."),

    ("ruthie08_cozy_helper",
     "Cozy, warm female voice assistant, early forties. Medium register, rounded "
     "and friendly. Gentle Midwest inflection, unhurried delivery. Sounds like "
     "she's got hot cocoa waiting. Helpful, patient, genuinely kind."),
]

# ============================================================================
# TEXAS
# ============================================================================

TRAVIS = [
    ("travis01_big_texas",
     "Deep male voice assistant with a Texas accent. Mid-forties. Big baritone, "
     "confident and warm. Natural Texan drawl, unhurried pacing. Sounds like a man "
     "who takes charge without raising his voice. Clear and authoritative."),

    ("travis02_ranch_steady",
     "Steady male voice assistant from Texas. Mid-forties. Deep baritone, warm and "
     "grounded. Light Texas drawl, deliberate delivery. Sounds capable and trustworthy. "
     "Takes his time, means what he says. Confident without being loud."),

    ("travis03_lone_star",
     "Confident male voice assistant with Texas character. Early forties. Rich "
     "baritone, warm and commanding. Natural drawl, measured pace. Sounds like he "
     "owns the room but will help you with anything. Deep, clear, genuine."),

    ("travis04_west_texas",
     "Male voice assistant with a West Texas drawl. Mid-forties. Deep, resonant "
     "baritone. Slow, confident delivery. Warm and straightforward. Sounds like a "
     "man who's never been in a hurry and never needed to be."),

    ("travis05_modern_texan",
     "Modern Texan male voice assistant, early forties. Deep mid-range, polished "
     "but with natural Texas inflection. Confident, clear delivery. Not a cowboy "
     "stereotype, just a capable man with a warm accent."),

    ("travis06_oil_baron",
     "Commanding male voice assistant from Texas. Mid-forties. Very deep baritone, "
     "rich and warm. Natural drawl, unhurried authority. Sounds like every word "
     "carries weight. Deep, steady, undeniable presence."),

    ("travis07_hill_country",
     "Warm male voice assistant, mid-forties. Deep baritone with a Hill Country Texas "
     "accent. Easy, confident pace. Clear and helpful without rushing. Sounds like "
     "a man who enjoys a good conversation. Grounded and real."),

    ("travis08_texas_oak",
     "Solid, deep male voice assistant from Texas, mid-forties. Rich baritone, "
     "steady and warm. Texas accent adds character without overwhelming. Measured "
     "pace, clear diction. An assistant who sounds unshakeable."),
]

BONNIE = [
    ("bonnie01_austin_quick",
     "Quick, witty female voice assistant from Austin. Late twenties. Mid-range, "
     "bright and sharp. Light modern Texas accent. Fast, helpful delivery with "
     "intelligence in every word. Not a stereotype, just genuinely quick."),

    ("bonnie02_modern_texas",
     "Modern Texan female voice assistant, late twenties. Mid-high register, clear "
     "and quick. Light Texas inflection, smart delivery. Sounds like she runs a "
     "startup and will also help you with your calendar. Sharp and warm."),

    ("bonnie03_quick_drawl",
     "Female voice assistant with a quick Texas drawl. Late twenties. Bright, "
     "medium register. Fast but clear delivery with light Southern warmth. "
     "Not slow, not sweet, just smart and helpful with Texas character."),

    ("bonnie04_lone_star_f",
     "Confident female voice assistant from Texas. Early thirties. Clear mid-range, "
     "steady with a light Texas accent. Direct, efficient delivery. Sounds capable "
     "and friendly. Modern Southern woman who gets things done."),

    ("bonnie05_austin_energy",
     "Energetic female voice assistant with Austin energy. Late twenties. Higher "
     "register, bright and fast. Light Texas warmth in the vowels. Sounds excited "
     "to help, genuinely engaged. Quick, clear, modern."),

    ("bonnie06_sharp_south",
     "Sharp, clear female voice assistant, late twenties. Mid-range, direct delivery "
     "with a light Texas accent. Efficient, smart, warm underneath. Not performatively "
     "Southern. Sounds like a woman who knows what she's doing."),

    ("bonnie07_warm_wit",
     "Warm, witty female voice assistant from Texas. Early thirties. Medium register, "
     "clear and friendly. Light drawl, quick pace. Sounds like she has a great sense "
     "of humor but keeps it professional. Helpful and genuine."),

    ("bonnie08_texas_bright",
     "Bright female voice assistant with Texas character. Late twenties. Clear, "
     "higher mid-range. Quick, articulate delivery with light Southern inflection. "
     "Sounds both smart and approachable. Modern Texas warmth."),
]

# ============================================================================
# BRITISH RP / LONDON
# ============================================================================

OLIVER = [
    ("oliver01_bbc_crisp",
     "Crisp, articulate male voice assistant with a British RP accent. Mid-thirties. "
     "Mid-range, clear and polished. Perfect received pronunciation, measured delivery. "
     "Sounds professional and intelligent. BBC newsreader quality."),

    ("oliver02_polished_brit",
     "Polished British male voice assistant, mid-thirties. Medium register, smooth "
     "RP accent. Clear, precise articulation, measured pace. Sounds effortlessly "
     "competent. Professional without being cold. Refined and helpful."),

    ("oliver03_westminster",
     "British male voice assistant with classic RP. Early forties. Mid-range baritone, "
     "clear and authoritative. Precise pronunciation, steady delivery. Sounds like he "
     "went to a good school and wants to help you with your files."),

    ("oliver04_news_desk",
     "Newsreader-quality British male voice assistant. Mid-thirties. Clear, bright "
     "mid-range. Impeccable RP accent, brisk but not rushed delivery. Sounds trustworthy "
     "and informed. Every word perfectly placed."),

    ("oliver05_subtle_posh",
     "Subtly posh British male voice assistant, mid-thirties. Medium register, smooth "
     "and clear. RP accent that's present but not overbearing. Sounds educated, warm, "
     "and genuinely helpful. Modern British professional."),

    ("oliver06_warm_rp",
     "Warm British male voice assistant with RP accent. Early forties. Mid-range, "
     "smooth and friendly. Precise pronunciation with natural warmth underneath. "
     "Not stuffy, just well-spoken. Clear, helpful, approachable."),

    ("oliver07_oxford_clear",
     "Clear, intelligent British male voice assistant. Mid-thirties. Medium register, "
     "crisp RP delivery. Sounds sharp and knowledgeable. Quick but measured pace. "
     "Professional, clean, trustworthy. Modern Oxford polish."),

    ("oliver08_classic_brit",
     "Classic British male voice assistant, mid-thirties. Clear mid-range baritone, "
     "received pronunciation. Steady, authoritative delivery with natural ease. "
     "Sounds like the most capable person in the room. Polished and genuine."),
]

PIPPA = [
    ("pippa01_posh_playful",
     "Posh but playful British female voice assistant. Late twenties. Higher register, "
     "bright and clear. RP accent with a sparkle. Quick, enthusiastic delivery. "
     "Sounds like she finds everything genuinely interesting. Fun and polished."),

    ("pippa02_brilliant_brit",
     "Bright British female voice assistant, late twenties. Higher mid-range, clear "
     "and enthusiastic. RP accent, quick pace. Everything sounds a bit exciting to her. "
     "Polished, warm, genuinely delighted to help."),

    ("pippa03_chelsea_charm",
     "Charming British female voice assistant, late twenties. Higher register, "
     "sparkling RP delivery. Quick, playful cadence. Posh but approachable. Sounds "
     "like she'd be fun at a party and also excellent at organizing it."),

    ("pippa04_crisp_warm",
     "Crisp, warm British female voice assistant. Early thirties. Mid-high register, "
     "clear RP accent. Friendly, efficient delivery. Polished without being distant. "
     "Sounds both competent and genuinely nice."),

    ("pippa05_bubbly_rp",
     "Bubbly British female voice assistant with RP accent. Late twenties. Bright, "
     "higher register. Quick, enthusiastic delivery. Sounds genuinely happy to help. "
     "Polished and playful at the same time."),

    ("pippa06_smart_sparkle",
     "Smart, sparkling British female voice assistant. Late twenties. Clear, higher "
     "mid-range. RP accent, quick cadence. Sounds sharp and engaged. Playful energy "
     "with real substance underneath. Helpful and bright."),

    ("pippa07_morning_posh",
     "Morning-bright British female voice assistant. Early thirties. Higher register, "
     "clear and warm. Light RP accent, friendly pace. Sounds fresh and ready to help. "
     "Polished but natural. Genuinely pleasant to listen to."),

    ("pippa08_kensington",
     "Kensington-polished female voice assistant, late twenties. Higher register, "
     "bright and clear. Refined RP with warmth. Quick, engaged delivery. Sounds "
     "posh in the best way. An assistant who makes everything sound manageable."),
]

ALFIE = [
    ("alfie01_east_london",
     "Cheeky male voice assistant with an East London accent. Late twenties. "
     "Mid-range, quick and lively. Cockney-influenced delivery with swagger. "
     "Sounds like he's having fun helping you. Clear, fast, characterful."),

    ("alfie02_cockney_charm",
     "Charming male voice assistant with a London accent. Late twenties. Medium "
     "register, quick delivery. East London inflection, crisp consonants. "
     "Sounds cheeky but genuinely helpful. Fast and engaging."),

    ("alfie03_borough_quick",
     "Quick male voice assistant from East London. Late twenties. Mid-range, "
     "bright and fast. London accent with character. Sounds sharp, street-smart, "
     "and eager to help. Punchy delivery, natural swagger."),

    ("alfie04_market_energy",
     "Energetic male voice assistant with London character. Early thirties. "
     "Mid-range, fast and engaging. East London accent, quick cadence. "
     "Sounds like he'd be great at selling anything, including being helpful."),

    ("alfie05_south_bank",
     "South Bank male voice assistant, late twenties. Mid-range, quick and clear. "
     "London accent with modern energy. Sounds young, sharp, and genuinely "
     "interested. Fast delivery with natural charm."),

    ("alfie06_cheeky_helper",
     "Cheeky, quick male voice assistant. Late twenties. Bright mid-range, "
     "London-accented delivery. Fast, fun, helpful. Sounds like he's enjoying "
     "the conversation. Cockney-lite with clear articulation."),

    ("alfie07_swagger_smart",
     "Smart male voice assistant with a bit of London swagger. Late twenties. "
     "Mid-range, confident and quick. East London accent, engaging delivery. "
     "Sounds both street-smart and book-smart. Helpful with personality."),

    ("alfie08_mile_end",
     "Mile End male voice assistant, late twenties. Mid-range, quick and "
     "characterful. London accent, punchy delivery. Sounds real and helpful. "
     "Not posh, not rough, just genuine East London with clarity."),
]

# ============================================================================
# BRITISH REGIONAL
# ============================================================================

ANGUS = [
    ("angus01_scottish_deep",
     "Deep male voice assistant with a Scottish accent. Mid-forties. Low baritone, "
     "resonant and warm. Rolling R's, distinct Scottish vowels. Measured, steady "
     "delivery. Sounds strong, reliable, and genuinely helpful."),

    ("angus02_highland_steady",
     "Steady Scottish male voice assistant, mid-forties. Deep baritone, warm and "
     "grounded. Highland accent, rolling consonants. Unhurried pace, clear delivery. "
     "Sounds like a man who could handle anything calmly."),

    ("angus03_edinburgh_warm",
     "Warm Scottish male voice assistant, early forties. Deep mid-range, smooth "
     "with a clear Scottish accent. Measured delivery, rolling R's. Sounds "
     "intelligent and dependable. Warm underneath the depth."),

    ("angus04_granite_voice",
     "Deep, granite-steady Scottish male voice assistant. Mid-forties. Bass-baritone, "
     "strong and resonant. Scottish accent, deliberate pacing. Sounds unflappable "
     "and trustworthy. A voice that inspires confidence."),

    ("angus05_scots_helper",
     "Helpful Scottish male voice assistant, mid-forties. Deep, warm baritone. "
     "Natural Scottish accent with rolling R's and distinct vowels. Steady pace, "
     "clear articulation. Friendly and reliable."),

    ("angus06_aberdeen_deep",
     "Deep, steady male voice assistant from Scotland. Mid-forties. Bass register, "
     "resonant and warm. Scottish accent, measured delivery. Sounds like he'd be "
     "calm in a blizzard. Strong, warm, dependable."),

    ("angus07_celtic_warm",
     "Warm Scottish male voice assistant, mid-forties. Deep, rich baritone with "
     "natural Scottish inflection. Rolling R's, clear vowels. Sounds both strong "
     "and kind. Unhurried, grounded, genuinely helpful."),

    ("angus08_north_sea",
     "North Sea male voice assistant, mid-forties. Deep, rough-edged baritone with "
     "a strong Scottish accent. Weathered but warm. Sounds like a man shaped by "
     "wind and rain. Steady, clear, reliable."),
]

RHYS = [
    ("rhys01_welsh_musical",
     "Male voice assistant with a musical Welsh accent. Early thirties. Mid-range, "
     "warm and melodic. Natural Welsh lilt, gentle rise and fall. Sounds friendly "
     "and genuine. Clear delivery with a singing quality."),

    ("rhys02_valleys_warm",
     "Warm Welsh male voice assistant, early thirties. Medium register, natural "
     "Welsh cadence. Musical inflection, warm delivery. Sounds like every sentence "
     "has a melody. Helpful, genuine, clear."),

    ("rhys03_cardiff_clear",
     "Clear Welsh male voice assistant, early thirties. Mid-range, bright and warm. "
     "Welsh accent with a natural lilt. Quick but melodic delivery. Sounds modern "
     "and helpful. Musical quality without being sing-song."),

    ("rhys04_lyrical_helper",
     "Lyrical male voice assistant with a Welsh accent. Early thirties. Medium "
     "register, warm and flowing. Natural Welsh musicality, gentle pacing. "
     "Sounds poetic and genuine. Clear, warm, engaging."),

    ("rhys05_cymru_steady",
     "Steady Welsh male voice assistant, early thirties. Mid-range, warm and even. "
     "Welsh accent, measured delivery. Musical undertone without affectation. "
     "Sounds grounded and genuinely helpful. Warm and reliable."),

    ("rhys06_pembroke_warm",
     "Warm male voice assistant from Wales, early thirties. Medium register, "
     "gentle and clear. Natural Welsh lilt, unhurried pace. Sounds kind and "
     "thoughtful. Musical quality, warm delivery."),

    ("rhys07_tenor_welsh",
     "Tenor-range Welsh male voice assistant, late twenties. Higher mid-range, "
     "bright and warm. Strong Welsh musicality, clear delivery. Sounds young, "
     "genuine, and eager to help. Melodic and clear."),

    ("rhys08_chapel_voice",
     "Rich Welsh male voice assistant, early thirties. Medium-low register, "
     "warm and resonant. Natural Welsh accent with a hymn-like quality. "
     "Measured, clear, helpful. Sounds both strong and gentle."),
]

GEMMA = [
    ("gemma01_manchester_warm",
     "Warm female voice assistant with a Manchester accent. Early thirties. "
     "Mid-range, bright and direct. Northern English delivery, clear and friendly. "
     "Sounds straight-shooting and genuinely helpful. No nonsense, all warmth."),

    ("gemma02_north_direct",
     "Direct female voice assistant from Northern England. Early thirties. Medium "
     "register, clear and quick. Manchester accent, efficient delivery. Sounds "
     "practical and warm. Straight to the point but friendly."),

    ("gemma03_lancashire_clear",
     "Clear female voice assistant with a Lancashire accent. Early thirties. "
     "Mid-range, bright and grounded. Northern warmth, direct delivery. Sounds "
     "capable and genuine. An assistant who tells it like it is."),

    ("gemma04_northern_star",
     "Northern English female voice assistant, early thirties. Medium register, "
     "warm and punchy. Manchester accent, quick cadence. Sounds smart, friendly, "
     "and no-nonsense. Efficient and likeable."),

    ("gemma05_cotton_mill",
     "Warm, grounded female voice assistant from Manchester. Early thirties. "
     "Mid-range, clear and strong. Northern accent, direct delivery. Sounds "
     "dependable and straight-talking. Warmth with substance."),

    ("gemma06_salford_quick",
     "Quick, friendly female voice assistant with a Salford accent. Early thirties. "
     "Mid-high register, bright and clear. Northern energy, fast delivery. "
     "Sounds helpful and genuine. Direct with a smile."),

    ("gemma07_steel_warm",
     "Warm but strong female voice assistant from Northern England. Early thirties. "
     "Medium register, clear and grounded. Manchester inflection, steady delivery. "
     "Sounds like she could run anything and would be nice about it."),

    ("gemma08_pennine_helper",
     "Friendly female voice assistant with Northern character. Early thirties. "
     "Mid-range, warm and clear. Manchester accent, measured pace. Sounds both "
     "approachable and competent. Genuine Northern warmth."),
]

# ============================================================================
# AUSTRALIAN
# ============================================================================

CALLUM = [
    ("callum01_sydney_laid",
     "Laid-back male voice assistant with an Australian accent. Early thirties. "
     "Mid-range, relaxed and warm. Natural Aussie inflection, easy pace. "
     "Sounds like nothing is a problem. Friendly, clear, helpful."),

    ("callum02_no_worries",
     "Easy-going Australian male voice assistant, early thirties. Medium register, "
     "warm and casual. Natural Aussie accent, relaxed delivery. Sounds genuinely "
     "relaxed and helpful. Clear despite the casual vibe."),

    ("callum03_bondi_clear",
     "Clear Australian male voice assistant, early thirties. Mid-range, bright "
     "and friendly. Aussie accent, natural pace. Sounds sunny and capable. "
     "Relaxed but articulate. Helpful with natural charm."),

    ("callum04_aussie_warm",
     "Warm Australian male voice assistant, early thirties. Medium register, "
     "natural and friendly. Aussie inflection, easy cadence. Sounds like a "
     "mate who also happens to be really helpful. Clear and genuine."),

    ("callum05_harbour_calm",
     "Calm Australian male voice assistant, early thirties. Mid-range, smooth "
     "and relaxed. Natural Aussie accent, measured delivery. Sounds grounded "
     "and capable. Nothing phases him. Clear and helpful."),

    ("callum06_outback_steady",
     "Steady Australian male voice assistant, early thirties. Medium baritone, "
     "warm and natural. Aussie accent, unhurried delivery. Sounds reliable "
     "and genuinely friendly. Clear, warm, easy to listen to."),

    ("callum07_coast_bright",
     "Bright Australian male voice assistant, late twenties. Mid-range, clear "
     "and energetic. Natural Aussie accent, quick but relaxed delivery. "
     "Sounds fresh and helpful. Sunny personality in the voice."),

    ("callum08_melbourne_cool",
     "Cool, collected Australian male voice assistant. Early thirties. Mid-range, "
     "smooth and clear. Aussie accent, measured pace. Sounds smart and laid-back. "
     "Effortlessly helpful. No rush."),
]

TESSA = [
    ("tessa01_melbourne_sharp",
     "Sharp female voice assistant with an Australian accent. Late twenties. "
     "Mid-range, clear and quick. Melbourne edge, dry delivery. Sounds smart "
     "and efficient. Helpful with a subtle wit underneath."),

    ("tessa02_dry_aussie",
     "Dry, quick Australian female voice assistant. Late twenties. Medium register, "
     "clear and direct. Melbourne accent, fast cadence. Sounds like she finds "
     "things amusing but will definitely help you. Sharp and warm."),

    ("tessa03_harbour_quick",
     "Quick Australian female voice assistant, late twenties. Mid-high register, "
     "bright and clear. Natural Aussie accent, efficient delivery. Sounds sharp "
     "and capable. Direct, helpful, a little dry."),

    ("tessa04_smart_aussie",
     "Smart Australian female voice assistant, late twenties. Clear mid-range, "
     "crisp delivery. Melbourne-modern accent, quick pace. Sounds like the "
     "smartest person in the room who also wants to help. Sharp and clean."),

    ("tessa05_southern_cross",
     "Clear, direct female voice assistant from Australia. Late twenties. "
     "Mid-range, bright and fast. Natural Aussie inflection, efficient. "
     "Sounds capable and genuine. Dry humor in the delivery."),

    ("tessa06_flat_white",
     "Cool, clear Australian female voice assistant. Late twenties. Medium "
     "register, smooth and direct. Melbourne accent, measured pace. Sounds "
     "modern and helpful. Understated personality, sharp delivery."),

    ("tessa07_quick_wit_oz",
     "Quick-witted Australian female voice assistant. Late twenties. Mid-range, "
     "bright and fast. Aussie accent, punchy delivery. Sounds fun and competent. "
     "Direct, helpful, with dry Australian humor."),

    ("tessa08_crisp_oz",
     "Crisp Australian female voice assistant, late twenties. Clear mid-high "
     "register, efficient delivery. Natural accent, clean articulation. "
     "Sounds professional and friendly. Quick, direct, helpful."),
]

# ============================================================================
# NEW ZEALAND
# ============================================================================

AROHA = [
    ("aroha01_kiwi_gentle",
     "Gentle female voice assistant with a New Zealand accent. Early thirties. "
     "Mid-range, warm and grounded. Natural Kiwi inflection, easy pace. "
     "Sounds genuinely kind and helpful. Calm, clear, centered."),

    ("aroha02_aotearoa_warm",
     "Warm Kiwi female voice assistant, early thirties. Medium register, smooth "
     "and gentle. New Zealand accent, unhurried delivery. Sounds grounded and "
     "nurturing. Helpful without being overbearing."),

    ("aroha03_wellington_calm",
     "Calm female voice assistant from New Zealand. Early thirties. Mid-range, "
     "clear and gentle. Kiwi accent, measured pace. Sounds centered and "
     "trustworthy. Warm, natural, genuinely helpful."),

    ("aroha04_pacific_gentle",
     "Gentle, grounded female voice assistant with a Kiwi accent. Early thirties. "
     "Medium register, warm and soft. Natural New Zealand inflection. Sounds "
     "like she has all the time in the world for you. Kind and clear."),

    ("aroha05_fern_steady",
     "Steady Kiwi female voice assistant, early thirties. Mid-range, warm and "
     "even. New Zealand accent, gentle delivery. Sounds dependable and genuine. "
     "Soft but clear. A calming, helpful presence."),

    ("aroha06_south_island",
     "Warm female voice assistant from New Zealand. Early thirties. Medium-low "
     "register, grounded and gentle. Kiwi accent, natural pace. Sounds earthy "
     "and real. Unhurried, warm, genuinely helpful."),

    ("aroha07_maori_warm",
     "Warm, centered female voice assistant with a New Zealand accent. Early "
     "thirties. Mid-range, smooth and gentle. Natural Kiwi inflection with a "
     "grounded quality. Sounds wise and kind. Clear, calm, helpful."),

    ("aroha08_ocean_gentle",
     "Gentle, ocean-calm female voice assistant from New Zealand. Early thirties. "
     "Medium register, soft and clear. Kiwi accent, easy pace. Sounds like "
     "peace and competence combined. Warm, grounded, lovely to listen to."),
]

# ============================================================================
# SOUTH AFRICAN
# ============================================================================

STELLAN = [
    ("stellan01_cape_town",
     "Male voice assistant with a South African accent. Early forties. Mid-range, "
     "clear and distinctive. Cape Town English, unique vowel sounds. Sounds worldly "
     "and intelligent. Clear delivery, warm and helpful."),

    ("stellan02_worldly_sa",
     "Worldly male voice assistant from South Africa. Early forties. Medium baritone, "
     "smooth and clear. Distinctive South African vowels. Measured delivery, warm "
     "and engaging. Sounds well-traveled and helpful."),

    ("stellan03_table_mountain",
     "Clear South African male voice assistant, early forties. Mid-range, distinctive "
     "accent. Clean, warm delivery. Sounds educated and genuine. Cape Town English "
     "with character. Helpful and reliable."),

    ("stellan04_safari_calm",
     "Calm South African male voice assistant, early forties. Medium register, warm "
     "and steady. Distinctive accent, measured pace. Sounds unflappable and kind. "
     "Clear, grounded, genuinely helpful."),

    ("stellan05_cape_clear",
     "Clear, articulate South African male voice assistant. Early forties. Mid-range, "
     "bright and warm. Cape Town accent, crisp delivery. Sounds professional and "
     "genuine. Modern, clean, helpful."),

    ("stellan06_garden_route",
     "Warm South African male voice assistant, early forties. Medium baritone, "
     "relaxed and clear. Distinctive vowels, easy pace. Sounds genuinely helpful "
     "and worldly. An interesting voice with real character."),

    ("stellan07_durban_warm",
     "Warm male voice assistant with South African character. Early forties. "
     "Mid-range, smooth and distinctive. Natural SA accent, friendly delivery. "
     "Sounds both exotic and approachable. Clear and helpful."),

    ("stellan08_jacaranda",
     "Distinctive South African male voice assistant, early forties. Medium "
     "register, clear and warm. Cape Town English, unique inflection. Sounds "
     "intelligent and genuine. A voice with character and warmth."),
]

# ============================================================================
# CANADIAN
# ============================================================================

MAPLE = [
    ("maple01_toronto_clean",
     "Clean, friendly female voice assistant with a Canadian accent. Late twenties. "
     "Mid-range, clear and neutral. Toronto-clean delivery, warm and helpful. "
     "Sounds approachable and professional. Subtle Canadian warmth."),

    ("maple02_friendly_north",
     "Friendly female voice assistant from Canada. Late twenties. Medium register, "
     "clear and bright. Neutral-ish Canadian accent, quick delivery. Sounds "
     "genuinely helpful and easy to talk to. Clean and warm."),

    ("maple03_polite_capable",
     "Politely capable female voice assistant, late twenties. Mid-range, clear "
     "and warm. Canadian English, clean delivery. Sounds both friendly and "
     "competent. Never pushy, always helpful."),

    ("maple04_queen_west",
     "Clear female voice assistant with Toronto character. Late twenties. "
     "Mid-high register, bright and friendly. Clean Canadian accent, quick "
     "pace. Sounds modern, smart, and genuinely nice."),

    ("maple05_northern_bright",
     "Bright Canadian female voice assistant, late twenties. Higher mid-range, "
     "clear and warm. Neutral accent, friendly delivery. Sounds like she's "
     "genuinely happy to help. Clean, quick, pleasant."),

    ("maple06_lakeside_calm",
     "Calm Canadian female voice assistant, late twenties. Medium register, "
     "smooth and clear. Clean accent, measured pace. Sounds both relaxed and "
     "capable. Warm without being slow. Genuinely helpful."),

    ("maple07_bilingual_warm",
     "Warm female voice assistant with a clean Canadian accent. Late twenties. "
     "Mid-range, friendly and clear. Neutral delivery with subtle warmth. "
     "Sounds cosmopolitan and approachable. Modern and helpful."),

    ("maple08_prairie_clear",
     "Clear, genuine female voice assistant from Canada. Late twenties. Medium "
     "register, clean and warm. Canadian English, natural delivery. Sounds "
     "trustworthy and friendly. Simple, clear, helpful."),
]

FRASER = [
    ("fraser01_alberta_outdoor",
     "Male voice assistant with an outdoorsy Canadian vibe. Early thirties. "
     "Mid-range, clear and steady. Slight Canadian raise, natural delivery. "
     "Sounds capable and grounded. Friendly, reliable, genuine."),

    ("fraser02_rockies_steady",
     "Steady Canadian male voice assistant, early thirties. Medium baritone, "
     "clear and warm. Slight Canadian inflection, measured pace. Sounds "
     "dependable and friendly. An assistant who feels solid."),

    ("fraser03_western_can",
     "Western Canadian male voice assistant, early thirties. Mid-range, clear "
     "and steady. Natural accent, unhurried delivery. Sounds grounded and "
     "capable. Warm, reliable, straightforward."),

    ("fraser04_banff_calm",
     "Calm Canadian male voice assistant, early thirties. Medium register, "
     "warm and steady. Clean accent with subtle Canadian character. Sounds "
     "unflappable and helpful. Clear, even, dependable."),

    ("fraser05_prairie_wind",
     "Clear, steady male voice assistant from Alberta. Early thirties. Mid-range, "
     "natural delivery. Canadian accent, moderate pace. Sounds honest and "
     "helpful. No frills, just genuine capability."),

    ("fraser06_north_country",
     "Northern Canadian male voice assistant, early thirties. Medium baritone, "
     "warm and grounded. Subtle Canadian inflection, steady delivery. Sounds "
     "like a man who knows what he's doing. Clear and helpful."),

    ("fraser07_chinook_warm",
     "Warm Canadian male voice assistant, early thirties. Mid-range, friendly "
     "and clear. Slight Albertan character, natural pace. Sounds genuine "
     "and helpful. Not flashy, just reliably good."),

    ("fraser08_lodge_steady",
     "Steady male voice assistant with Canadian character. Early thirties. "
     "Medium register, warm and clear. Natural accent, measured delivery. "
     "Sounds like someone you'd want guiding you. Calm and capable."),
]

# ============================================================================
# WILD CARDS
# ============================================================================

KIT = [
    ("kit01_androgynous_clean",
     "Clean, androgynous voice assistant. Late twenties. Mid-range, neither "
     "distinctly male nor female. Clear, neutral delivery. Mid-Atlantic polish. "
     "Sounds modern, smart, and ungendered. Helpful and clean."),

    ("kit02_neutral_bright",
     "Bright, gender-neutral voice assistant. Late twenties. Higher mid-range, "
     "clear and clean. No strong gender markers. Mid-Atlantic clarity, quick "
     "delivery. Sounds fresh, modern, and genuinely helpful."),

    ("kit03_crystal_clear",
     "Crystal-clear androgynous voice assistant. Late twenties. Medium register, "
     "perfectly clean. Could be anyone. Neutral accent, efficient delivery. "
     "Sounds focused and helpful. Pure clarity."),

    ("kit04_glass_smooth",
     "Smooth, gender-neutral voice assistant. Late twenties. Mid-range, even "
     "and clear. No accent, no gender lean. Sounds like the platonic ideal "
     "of a helpful voice. Clean, warm, modern."),

    ("kit05_modern_neutral",
     "Modern, neutral voice assistant. Late twenties. Mid-range, clear and "
     "balanced. Neither masculine nor feminine. Clean delivery, helpful tone. "
     "Sounds like the future of voice assistants. Natural and clean."),

    ("kit06_silk_mid",
     "Silky, mid-range voice assistant with no gender markers. Late twenties. "
     "Smooth, clear, androgynous. Mid-Atlantic clarity, measured pace. "
     "Sounds polished and genuinely helpful. Clean and modern."),

    ("kit07_mercury_voice",
     "Fluid, androgynous voice assistant. Late twenties. Clear mid-range, "
     "balanced timbre. No strong accent, no gender lean. Quick, helpful "
     "delivery. Sounds like pure competence. Modern and clean."),

    ("kit08_prism_clear",
     "Clear, balanced voice assistant. Late twenties. Mid-range, androgynous, "
     "clean. Neutral delivery with natural warmth. Could be anyone's voice. "
     "Sounds both familiar and unique. Helpful and genuine."),
]

WREN = [
    ("wren01_whisper_warm",
     "Whispery, intimate voice assistant. Late twenties. Soft, low register. "
     "Close-mic feel, gentle delivery. ASMR-adjacent warmth without affectation. "
     "Sounds calming and helpful. Soft but clear."),

    ("wren02_asmr_helper",
     "Soft, intimate voice assistant. Late twenties. Low, whispery register. "
     "Gentle, measured delivery. Sounds like a late-night radio host. "
     "Calming, clear despite the softness. Soothing and helpful."),

    ("wren03_velvet_whisper",
     "Velvety, soft voice assistant. Late twenties. Low register, intimate "
     "delivery. Breathy but clear. Sounds like a warm secret. ASMR quality "
     "without trying. Gentle, calming, helpful."),

    ("wren04_night_soft",
     "Night-soft voice assistant. Late twenties. Very low, gentle register. "
     "Intimate, whispery delivery. Sounds like 3am comfort. Soft, warm, "
     "and still perfectly clear. Calming presence."),

    ("wren05_close_gentle",
     "Close, gentle voice assistant. Late twenties. Soft mid-range, intimate. "
     "Whisper-adjacent but articulate. Sounds personal and caring. "
     "Gentle delivery, clear words. Warm and soothing."),

    ("wren06_feather_light",
     "Feather-light voice assistant. Late twenties. Soft, airy register. "
     "Gentle, intimate delivery. Sounds like a whispered recommendation. "
     "Clear despite the softness. Calming and helpful."),

    ("wren07_silk_whisper",
     "Silky, whispery voice assistant. Late twenties. Low, smooth register. "
     "Intimate delivery with perfect clarity. Sounds like luxury for your "
     "ears. Soft, warm, genuinely helpful."),

    ("wren08_moonlight",
     "Moonlight-soft voice assistant. Late twenties. Low register, gentle "
     "and clear. Intimate, unhurried delivery. Sounds peaceful and competent. "
     "A whisper that somehow carries authority."),
]

KNOX = [
    ("knox01_deep_narrator",
     "Deep, authoritative male voice assistant. Mid-forties. Very deep bass, "
     "rich and resonant. Audiobook narrator quality. Measured, commanding "
     "delivery. Sounds like every word matters. Clear and powerful."),

    ("knox02_bass_command",
     "Commanding deep male voice assistant, mid-forties. True bass register, "
     "smooth and authoritative. Measured pace, impeccable delivery. Sounds "
     "like a narrator of important things. Rich, deep, trustworthy."),

    ("knox03_velvet_bass",
     "Velvet-bass male voice assistant, mid-forties. Very deep, smooth and "
     "rich. Authoritative but warm. Narrator-quality delivery. Sounds like "
     "he should be voicing documentaries. Deep, clear, helpful."),

    ("knox04_documentary",
     "Documentary-voice male assistant, mid-forties. Deep bass-baritone, "
     "rich and clear. Authoritative, measured delivery. Sounds like David "
     "Attenborough's helpful nephew. Deep, warm, commanding."),

    ("knox05_broadcast_deep",
     "Broadcast-quality deep male voice assistant. Mid-forties. Bass register, "
     "perfectly clear. Authoritative, steady delivery. Sounds like the most "
     "trustworthy voice on earth. Rich, resonant, helpful."),

    ("knox06_midnight_bass",
     "Midnight-deep male voice assistant, mid-forties. Very deep bass, warm "
     "and resonant. Slow, measured delivery. Sounds like comfort and authority "
     "combined. An assistant who makes everything sound important."),

    ("knox07_theater_deep",
     "Theater-trained deep male voice assistant. Mid-forties. Rich bass, "
     "projecting and clear. Measured, deliberate delivery. Sounds both "
     "powerful and approachable. Narrator quality in an assistant."),

    ("knox08_oak_bass",
     "Oak-deep male voice assistant, mid-forties. True bass, resonant and "
     "steady. Clean, authoritative delivery. Sounds immovable and warm. "
     "An assistant who could narrate your life."),
]

PEARL = [
    ("pearl01_grandmother_wise",
     "Elderly female voice assistant, mid-seventies. Higher register, thin "
     "but warm. Slight Southern inflection, slow and patient delivery. "
     "Sounds like a wise grandmother helping you. Gentle, clear, kind."),

    ("pearl02_silver_south",
     "Silver-voiced elderly female assistant, mid-seventies. Thinner register, "
     "warm and wise. Light Southern accent, unhurried pace. Sounds like she's "
     "seen everything and still wants to help. Patient and genuine."),

    ("pearl03_porch_wisdom",
     "Wise, elderly female voice assistant, mid-seventies. Soft, thin voice "
     "with Southern warmth. Slow, deliberate delivery. Sounds like advice from "
     "someone who's been right many times. Gentle and clear."),

    ("pearl04_quilting_bee",
     "Warm elderly female voice assistant, mid-seventies. Higher, thinner "
     "register. Light Southern accent, patient delivery. Sounds like a "
     "grandmother who always has time for you. Kind, clear, steady."),

    ("pearl05_magnolia_old",
     "Elderly female voice assistant with Southern grace. Mid-seventies. "
     "Soft, warm, slightly wavering. Slow, measured delivery. Sounds like "
     "wisdom and warmth combined. Patient, gentle, genuine."),

    ("pearl06_sweet_tea",
     "Sweet, elderly female voice assistant, mid-seventies. Higher register, "
     "warm and gentle. Light Southern drawl, unhurried. Sounds comforting "
     "and wise. An assistant who makes everything feel manageable."),

    ("pearl07_rocking_chair",
     "Gentle elderly female voice assistant, mid-seventies. Soft, thin voice "
     "with character. Southern warmth, slow pace. Sounds like she's helping "
     "from a rocking chair. Patient, kind, clear."),

    ("pearl08_heritage_warm",
     "Warm, wise elderly female voice assistant. Mid-seventies. Thinner "
     "register, gentle delivery. Light Southern accent, measured pace. "
     "Sounds like family wisdom. Kind, patient, genuinely helpful."),
]

ZIGGY = [
    ("ziggy01_high_energy",
     "High-energy voice assistant, early twenties. Bright, fast, excited. "
     "Higher register, quick cadence. Sounds like a podcast host who's "
     "genuinely thrilled about everything. Clear, punchy, infectious energy."),

    ("ziggy02_podcast_fire",
     "Podcast-host-energy voice assistant, early twenties. Bright and fast. "
     "Higher register, rapid delivery. Sounds like they can barely contain "
     "their excitement about helping. Energetic, clear, fun."),

    ("ziggy03_caffeine_bright",
     "Caffeine-bright voice assistant, early twenties. Very bright register, "
     "fast and clear. Sounds like three espressos and genuine enthusiasm. "
     "Quick, punchy, helpful. Energy without being annoying."),

    ("ziggy04_sunrise_pop",
     "Sunrise-energy voice assistant, early twenties. Bright, higher register. "
     "Quick, bouncy delivery. Sounds like the most awake person alive. "
     "Enthusiastic, clear, genuinely fun to listen to."),

    ("ziggy05_turbo_helper",
     "Turbo-charged voice assistant, early twenties. Bright and fast. Higher "
     "register, rapid cadence. Sounds excited about every task. Clear "
     "despite the speed. Infectious energy, genuinely helpful."),

    ("ziggy06_neon_bright",
     "Neon-bright voice assistant, early twenties. Very bright register, "
     "quick and punchy. Sounds young, excited, and genuinely helpful. "
     "Fast delivery, clear words. Pure energy."),

    ("ziggy07_spark_quick",
     "Spark-quick voice assistant, early twenties. Bright, fast, higher "
     "register. Quick delivery with genuine warmth. Sounds like they live "
     "for helping people. Energetic, clear, fun."),

    ("ziggy08_electric_pop",
     "Electric-energy voice assistant, early twenties. Bright, fast, clear. "
     "Higher register, bouncy delivery. Sounds like excitement personified. "
     "Quick, helpful, impossible not to smile at."),
]

# ============================================================================
# ALL VOICES
# ============================================================================

ALL_VOICES = {
    "vinnie": VINNIE,
    "sage": SAGE,
    "dakota": DAKOTA,
    "bryce": BRYCE,
    "clyde": CLYDE,
    "ruthie": RUTHIE,
    "travis": TRAVIS,
    "bonnie": BONNIE,
    "oliver": OLIVER,
    "pippa": PIPPA,
    "alfie": ALFIE,
    "angus": ANGUS,
    "rhys": RHYS,
    "gemma": GEMMA,
    "callum": CALLUM,
    "tessa": TESSA,
    "aroha": AROHA,
    "stellan": STELLAN,
    "maple": MAPLE,
    "fraser": FRASER,
    "kit": KIT,
    "wren": WREN,
    "knox": KNOX,
    "pearl": PEARL,
    "ziggy": ZIGGY,
}

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)

total = sum(len(c) for c in ALL_VOICES.values())
print(f"Generating {total} candidates across {len(ALL_VOICES)} voices\n")

all_results = []
for voice_name, candidates in ALL_VOICES.items():
    print(f"\n{'='*60}")
    print(f" {voice_name.upper()} — {len(candidates)} candidates")
    print(f"{'='*60}\n")

    for name, instruct in candidates:
        print(f"-> {name}", flush=True)
        t = time.time()
        chunks = []
        for result in model.generate_voice_design(
            text=TARGET_TEXT,
            instruct=instruct,
            language="english",
            temperature=0.9,
            stream=False,
        ):
            chunks.append(result.audio)

        if not chunks:
            print("   !! no audio chunks produced", flush=True)
            continue

        audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
        sr = 24000
        out_path = OUT_DIR / f"{name}.wav"
        sf.write(str(out_path), audio, sr)
        dur = len(audio) / sr
        elapsed = time.time() - t
        print(f"   done {dur:.1f}s audio, gen {elapsed:.1f}s", flush=True)
        all_results.append((name, instruct, out_path, dur))

index_path = OUT_DIR / "index.txt"
with open(index_path, "w") as f:
    f.write(f"Voice candidates — generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Target text: {TARGET_TEXT}\n\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in all_results:
        f.write(f"\n{name}  ({dur:.1f}s)\n")
        f.write(f"  file: {path.name}\n")
        f.write(f"  instruct: {instruct}\n")

print(f"\nWrote index to {index_path}")
print(f"ALL DONE — {len(all_results)} candidates in {OUT_DIR}")
