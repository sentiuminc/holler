"""Batch 2: Generate more training data. Continues numbering from batch 1."""
import time
import os
import json
import numpy as np
import soundfile as sf
import mlx.core as mx

REF_AUDIO = "/Users/nagy/Downloads/joe-candidates/joe08h_subtle_nudge.wav"
REF_TEXT = "Good morning. I've reviewed your schedule for today, and everything is in order. Your first meeting begins in forty-five minutes. Would you like me to prepare the car?"
OUTPUT_DIR = os.path.expanduser("~/Downloads/joe-training-data")
JSONL_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")

# Count existing clips to continue numbering
existing = len([f for f in os.listdir(os.path.join(OUTPUT_DIR, "audio")) if f.endswith(".wav")])
print(f"Existing clips: {existing}. New clips will start at {existing + 1}.")
START_IDX = existing

TEXTS = [
    # Nature and the world
    "The ocean covers about seventy percent of the Earth's surface, and we've only explored a tiny fraction of it.",
    "There's something about watching a sunset that makes everything else feel less urgent.",
    "Did you know that honey never spoils? Archaeologists have found three thousand year old honey that was still perfectly edible.",
    "Mountains form over millions of years as tectonic plates push against each other. It's geology in slow motion.",
    "The Amazon rainforest produces roughly twenty percent of the world's oxygen. People call it the lungs of the Earth.",
    "Lightning strikes the Earth about eight million times a day. Most of it happens over tropical regions.",
    "The deepest point in the ocean is the Mariana Trench, at nearly eleven kilometers below the surface.",
    "Autumn leaves change color because the chlorophyll breaks down, revealing the yellow and orange pigments underneath.",
    "The Northern Lights happen when charged particles from the sun interact with gases in our atmosphere.",
    "A single tree can absorb about twenty two kilograms of carbon dioxide per year.",
    "Coral reefs support about twenty five percent of all marine species, even though they cover less than one percent of the ocean floor.",
    "The Sahara Desert isn't just sand. It has mountains, oases, and even the occasional snowfall.",
    "Wolves were reintroduced to Yellowstone in the nineties, and it completely transformed the ecosystem.",
    "The oldest known living tree is a bristlecone pine in California. It's over four thousand years old.",
    "Earthquakes happen because the Earth's crust is made up of plates that are constantly moving, even if we can't feel it.",

    # Science and space
    "Light from the sun takes about eight minutes to reach Earth. So when you look at the sun, you're seeing it as it was eight minutes ago.",
    "The human brain uses about twenty percent of the body's total energy, even though it only weighs about one point four kilograms.",
    "There are more stars in the universe than grains of sand on all of Earth's beaches. It's almost impossible to comprehend.",
    "Water is the only common substance that exists naturally in all three states: solid, liquid, and gas.",
    "Your body replaces most of its cells over a seven to ten year period. You're literally not the same person you were a decade ago.",
    "The speed of sound is about three hundred and forty three meters per second. That's why you see lightning before you hear thunder.",
    "Black holes are so dense that not even light can escape their gravitational pull. They bend space and time around them.",
    "The human eye can distinguish roughly ten million different colors. Most of the processing happens in the brain, not the eye itself.",
    "DNA contains the instructions for building every protein in your body. If you stretched it all out, it would reach the sun and back several times.",
    "Gravity on the moon is about one sixth of what it is on Earth. That's why astronauts can bounce around so easily up there.",
    "The International Space Station orbits Earth about sixteen times every day, traveling at roughly twenty eight thousand kilometers per hour.",
    "Octopuses have three hearts and blue blood. Two hearts pump blood to the gills, and one pumps it to the rest of the body.",
    "A teaspoon of neutron star material would weigh about six billion tons on Earth.",
    "The human body contains enough iron to make a small nail. Not much, but it's essential for carrying oxygen in the blood.",
    "Quantum entanglement means two particles can be connected in a way that measuring one instantly affects the other, no matter the distance.",

    # History and culture
    "The printing press changed everything. Before Gutenberg, books had to be copied by hand, one page at a time.",
    "Ancient Romans used concrete that was in some ways better than what we use today. Their structures have lasted two thousand years.",
    "The Great Wall of China isn't actually visible from space with the naked eye. That's one of those myths that won't go away.",
    "Coffee was discovered in Ethiopia, according to legend, when a goat herder noticed his animals getting energetic after eating certain berries.",
    "The library of Alexandria was one of the greatest centers of knowledge in the ancient world. Its destruction is still debated by historians.",
    "Chess has been around for over fifteen hundred years. It originated in India and spread along trade routes to Persia and then Europe.",
    "The first email was sent in nineteen seventy one. The content of the message has been forgotten, probably something like a test.",
    "Paper money was first used in China during the Tang dynasty, around the seventh century. Europe didn't adopt it until much later.",
    "The Rosetta Stone was the key to deciphering Egyptian hieroglyphics. It contained the same text in three different scripts.",
    "Viking explorers reached North America about five hundred years before Columbus. They called it Vinland.",
    "The concept of zero as a number was developed independently in several cultures, but Indian mathematicians formalized it around the fifth century.",
    "Shakespeare invented over seventeen hundred words that we still use today, including lonely, generous, and eyeball.",
    "The Silk Road wasn't a single road. It was a network of trade routes connecting East Asia to the Mediterranean.",
    "Cleopatra lived closer in time to the moon landing than to the building of the Great Pyramid.",
    "The invention of the telegraph in the eighteen thirties was the first time humans could communicate instantly over long distances.",

    # Food and cooking
    "The secret to a good pasta is salting the water generously. It should taste like the sea.",
    "Sourdough bread takes patience. The fermentation process can take anywhere from twelve to twenty four hours.",
    "Sushi rice is seasoned with a mixture of rice vinegar, sugar, and salt. Getting the balance right is what makes it special.",
    "Dark chocolate is actually good for you in moderation. It's rich in antioxidants and can improve blood flow.",
    "The best scrambled eggs are cooked low and slow. High heat makes them rubbery.",
    "Fermentation is one of the oldest food preservation techniques. It's how we get cheese, wine, yogurt, and kimchi.",
    "A perfectly ripe avocado should give slightly when you press it. Too soft means it's overripe.",
    "Cast iron pans get better with age. The seasoning builds up over time and creates a natural nonstick surface.",
    "Fresh herbs make a huge difference in cooking. Adding them at the end preserves their flavor and color.",
    "The maillard reaction is what gives browned food its complex flavor. It's not just caramelization, it's a chemical reaction between amino acids and sugars.",

    # Daily life and observations
    "There's a specific kind of tiredness that comes from being productive all day. It feels different from doing nothing.",
    "Sometimes the best part of the day is that first cup of coffee in the morning, when everything is still quiet.",
    "People underestimate how much a good night's sleep can change your perspective on a problem.",
    "The sound of rain on a window is one of those universally calming things. There's probably an evolutionary reason for it.",
    "Walking is underrated as exercise. A thirty minute walk every day can do wonders for both physical and mental health.",
    "There's always that one drawer in every house that becomes the catch all for random stuff nobody knows where to put.",
    "The smell of freshly baked bread is almost universally liked. Some real estate agents use it during open houses.",
    "Handwritten notes feel more personal than text messages. There's something about the effort that matters.",
    "People tend to overestimate what they can do in a day and underestimate what they can do in a year.",
    "The best conversations usually happen when you're not trying to have a great conversation. They just flow naturally.",
    "There's a word in Japanese, komorebi, that means sunlight filtering through tree leaves. Some concepts just don't translate easily.",
    "Most people check their phone within the first ten minutes of waking up. It's become such an automatic habit.",
    "The difference between a house and a home is entirely about how it makes you feel when you walk through the door.",
    "Libraries are one of the few remaining public spaces where you can just exist without being expected to buy something.",
    "A clean desk doesn't necessarily mean a productive person, but it does tend to reduce mental clutter.",

    # Travel
    "Tokyo at night feels like stepping into the future. The neon lights, the efficiency, the energy of the city.",
    "There's a small town in Iceland where the entire population is less than three hundred, and they still have a swimming pool.",
    "The best travel advice is to pack half of what you think you need and bring twice the money.",
    "Night trains across Europe have their own kind of romance. Falling asleep in one country and waking up in another.",
    "Street food in Bangkok is some of the best food you'll ever eat. And it costs almost nothing.",
    "The difference between traveling and tourism is how much you're willing to get lost.",
    "Some of the most beautiful places in the world are the hardest to get to. That's probably not a coincidence.",
    "Jet lag is your body's way of reminding you that you just crossed several time zones in a metal tube at nine hundred kilometers per hour.",
    "Learning even ten words of the local language can completely change how people treat you when you travel.",
    "The best souvenirs aren't objects. They're stories and memories that you carry with you.",

    # Music and art
    "Music has this incredible ability to transport you back to a specific moment in time. Just a few notes can trigger a flood of memories.",
    "The piano has eighty eight keys, but the number of possible combinations is essentially infinite.",
    "Abstract art isn't about what it looks like. It's about what it makes you feel. That's what makes it hard to judge objectively.",
    "The best album covers become inseparable from the music itself. You can't think of one without the other.",
    "Jazz is about the notes you don't play as much as the ones you do. The silence between phrases is where the magic happens.",
    "Photography changed how humans see the world. For the first time, you could freeze a moment exactly as it happened.",
    "There's a theory that all stories follow one of about seven basic plot structures. Everything else is variation.",
    "The best designs are invisible. You only notice design when it's bad.",
    "Animation is the art of making static images feel alive. It takes about twelve to twenty four frames per second to create the illusion of motion.",
    "Every font tells a story before you even read the words. Typography is one of those things most people never consciously think about.",

    # Technology and society
    "The internet was originally designed to survive a nuclear war. Now we use it to share pictures of our lunch.",
    "Moore's law predicted that computing power would double roughly every two years. It held true for decades.",
    "The first computer mouse was made of wood. It was invented in nineteen sixty four by Douglas Engelbart.",
    "Email was supposed to make communication faster. Instead, it created an entirely new category of stress.",
    "Social media algorithms are designed to maximize engagement, not happiness. Those are very different objectives.",
    "The average person spends about seven hours a day looking at screens. That's almost half of our waking life.",
    "Self driving cars have to solve philosophical problems that humans handle instinctively. The trolley problem is actually relevant now.",
    "Cloud computing basically means using someone else's computer. The metaphor makes it sound more magical than it is.",
    "The first website ever created is still online. It's a plain text page about the World Wide Web project.",
    "Passwords are a terrible security mechanism, but we haven't found a replacement that works as universally.",
    "Three D printing started in the eighties but only became accessible in the last decade. Some people print entire houses now.",
    "Streaming services have more content than anyone could watch in a lifetime. The paradox of choice is real.",
    "Open source software powers most of the internet. The biggest projects are maintained by volunteers.",
    "Video games generate more revenue than movies and music combined. It's the largest entertainment industry in the world.",
    "The first text message was sent in nineteen ninety two. It said Merry Christmas.",

    # Philosophy and thinking
    "The only constant is change. That's been true for thousands of years, which is kind of ironic.",
    "We spend so much time planning for the future that we forget to pay attention to what's happening right now.",
    "Mistakes are just data points. They tell you what doesn't work, which is genuinely valuable information.",
    "The map is not the territory. Our models of reality are always simpler than reality itself.",
    "Most of the things people worry about never actually happen. But telling someone not to worry doesn't help.",
    "Creativity isn't about having original ideas. It's about making unexpected connections between existing ones.",
    "The difference between knowledge and wisdom is that knowledge is knowing what to do, and wisdom is knowing when to do it.",
    "Sometimes the hardest part of solving a problem is admitting that it exists in the first place.",
    "Perfection is the enemy of done. At some point you have to ship it and iterate.",
    "The best way to learn something is to try to teach it to someone else. It exposes all the gaps in your understanding.",
    "Confidence isn't about knowing you're right. It's about being comfortable with being wrong.",
    "Simple doesn't mean easy. Making something simple often requires more effort than making something complex.",
    "The stories we tell ourselves about who we are have a bigger impact on our behavior than objective reality does.",
    "Asking the right question is often more important than having the right answer.",
    "Progress isn't always visible. Some of the most important changes happen slowly, beneath the surface.",

    # Sports and fitness
    "Running is one of those activities that gets easier the more you do it, but the first few weeks are brutal.",
    "The difference between a good athlete and a great one is usually mental, not physical.",
    "Stretching after exercise is just as important as the exercise itself. Most people skip it though.",
    "Swimming works every major muscle group in your body. It's also one of the lowest impact exercises you can do.",
    "Professional athletes spend way more time training than competing. The game is just the tip of the iceberg.",
    "Rest days aren't laziness. They're when your body actually repairs and gets stronger.",
    "Rock climbing is as much about problem solving as it is about strength. Each route is like a puzzle.",
    "The marathon distance of twenty six point two miles was standardized at the nineteen oh eight Olympics in London.",
    "Good posture isn't just about sitting up straight. It's about how you distribute weight across your entire skeleton.",
    "Breathing technique is the most underrated aspect of any physical activity. It affects everything else.",

    # Animals
    "Elephants are one of the few animals that can recognize themselves in a mirror. They also mourn their dead.",
    "Cats spend about seventy percent of their lives sleeping. That works out to roughly sixteen hours a day.",
    "Bees communicate the location of flowers by doing a specific dance. It's one of the most sophisticated forms of animal communication.",
    "Dolphins sleep with one eye open and half their brain awake. They need to stay conscious enough to breathe.",
    "A group of flamingos is called a flamboyance. That might be the most appropriate collective noun in the English language.",
    "Dogs have about three hundred million scent receptors in their noses, compared to about six million in humans.",
    "Crows can recognize individual human faces and will remember if someone treated them well or badly.",
    "The mantis shrimp can see colors that humans can't even imagine. They have sixteen types of color receptors. We have three.",
    "Penguins propose to their partners with a pebble. If the other penguin accepts it, they become a pair.",
    "Tardigrades can survive in the vacuum of space, extreme radiation, and temperatures close to absolute zero.",

    # Random interesting facts
    "The average person walks about a hundred thousand miles in their lifetime. That's roughly four trips around the Earth.",
    "Bananas are technically berries, but strawberries aren't. Botanical classifications don't always match common sense.",
    "The shortest war in history lasted thirty eight minutes. It was between Britain and Zanzibar in eighteen ninety six.",
    "There are more possible chess games than atoms in the observable universe.",
    "Honey bees must visit about two million flowers to make one pound of honey.",
    "A cloud can weigh more than a million pounds. They float because the water droplets are spread over a huge area.",
    "The average person spends two weeks of their lifetime waiting for traffic lights to change.",
    "Octopuses have nine brains. One central brain and a smaller one in each of their eight arms.",
    "The dot over the letter i is called a tittle.",
    "Finland has more saunas than cars. There are roughly three million saunas for a population of five and a half million.",

    # Casual conversation and reactions
    "Oh, that's wild. I did not see that coming at all.",
    "Yeah, I mean, it's not ideal, but it's definitely workable.",
    "Huh, I never really thought about it that way before.",
    "Okay, that actually explains a lot.",
    "Wait, really? That's way simpler than I expected.",
    "I mean, you're not wrong. It's just a different way of looking at it.",
    "That's kind of what I figured, but it's good to have it confirmed.",
    "No way. That's actually hilarious.",
    "So basically what you're saying is we overthought this.",
    "Alright, I'm sold. Let's go with that.",
    "Hmm, I'm like fifty fifty on that one.",
    "Oh come on, that's not even close to what happened.",
    "Sure, but have you considered the other side of it?",
    "That's the kind of thing that sounds crazy until you try it.",
    "Honestly? I have no idea. But I know someone who might.",

    # Weather and seasons
    "There's something about the first warm day of spring that makes everyone a little bit happier.",
    "Snow has a way of making everything quiet. The flakes absorb sound, so the world literally gets softer.",
    "Thunderstorms are terrifying and beautiful at the same time. The power of nature is hard to ignore.",
    "The best kind of weather is the kind where you don't have to think about what to wear.",
    "Wind chill makes cold temperatures feel even colder because it strips heat from your skin faster.",
    "Some people love the rain. Others hate it. Very few people feel neutral about it.",
    "Fog is basically a cloud that touched the ground. It forms when air cools below its dew point.",
    "Summer evenings that stay warm late into the night are one of life's simple pleasures.",
    "The longest day of the year happens in June in the northern hemisphere. After that, the days slowly get shorter.",
    "Rainbows are actually full circles, but we can only see a semicircle because the ground gets in the way.",

    # Books and reading
    "A good book can change how you think about the world. That's a lot of power for a few hundred pages.",
    "Audiobooks aren't cheating. Your brain processes the story the same way whether you read it or hear it.",
    "Some books are better the second time you read them because you notice things you missed the first time.",
    "The smell of old books comes from the chemical breakdown of compounds in the paper and ink. It even has a name. Bibliosmia.",
    "Reading before bed helps your brain transition to sleep mode. Screens do the opposite.",
    "Short stories are an underrated art form. Telling a complete story in a few pages requires incredible skill.",
    "Public libraries lend more than books now. Some lend tools, musical instruments, even fishing equipment.",
    "The most translated book in the world is The Little Prince by Antoine de Saint-Exupery.",
    "Speed reading is mostly a myth. Comprehension drops significantly when you read faster than about three hundred words per minute.",
    "Used bookstores have a charm that new ones can't replicate. Every book on the shelf has its own history.",

    # Work and productivity
    "The most productive hour of the day varies from person to person. Some people peak at six AM, others at midnight.",
    "Multitasking is a myth for most cognitive tasks. Your brain actually switches between tasks, and each switch costs you time.",
    "Taking breaks actually improves productivity. Your brain needs time to consolidate information.",
    "The two minute rule says if something takes less than two minutes, do it now instead of adding it to your list.",
    "Most meetings could have been an email. That's not a joke. It's a widely documented productivity problem.",
    "Deep work requires at least twenty to thirty minutes of uninterrupted focus to get into a flow state.",
    "Writing things down helps you remember them, even if you never look at the notes again. The act of writing itself is what matters.",
    "Deadlines can be motivating, but artificial urgency on everything makes real urgency invisible.",
    "The best time to tackle your hardest task is first thing in the morning, before decision fatigue sets in.",
    "Inbox zero is satisfying but not necessary. What matters is that you respond to the things that actually need a response.",

    # Relationships and people
    "The best friendships are the ones where you can pick up right where you left off, even after months of not talking.",
    "Listening is the most underrated skill in communication. Most people are just waiting for their turn to speak.",
    "Small acts of kindness compound over time. They might seem insignificant in the moment, but they add up.",
    "People remember how you made them feel long after they forget what you said.",
    "The hardest part of any disagreement is genuinely trying to understand the other person's perspective.",
    "Trust is built in drops and lost in buckets. It takes years to establish and seconds to destroy.",
    "Everyone you meet knows something you don't. That's a good reason to stay curious about other people.",
    "Saying no is a skill that most people struggle with, but it's essential for protecting your time and energy.",
    "The people who have the biggest impact on your life often don't realize it.",
    "Gratitude isn't just polite. Studies show it literally changes your brain chemistry and makes you happier.",

    # Misc short and punchy
    "That's the real question, isn't it?",
    "Well, there you go.",
    "Makes sense to me.",
    "Works for me.",
    "I'll figure it out.",
    "We'll see how it goes.",
    "Not bad at all.",
    "Could be worse.",
    "Better than expected.",
    "Noted.",
    "Let's move on.",
    "That tracks.",
    "I appreciate that.",
    "Right on.",
    "Sounds about right.",
    "Can't argue with that.",
    "Point taken.",
    "Message received.",
    "Consider it done.",
    "On it.",
]

print(f"Batch 2 texts: {len(TEXTS)}")
print(f"Loading Qwen3-TTS 1.7B Base (8-bit)...")
t0 = time.time()
from mlx_audio.tts import load
model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
print(f"Model loaded in {time.time()-t0:.1f}s")

print("Warmup pass...")
for _ in model.generate(text="Hello.", ref_audio=REF_AUDIO, ref_text=REF_TEXT, language="en", temperature=0.6):
    pass
print("Warmup done.\n")

silence_1s = np.zeros(24000, dtype=np.float32)
total_audio_duration = 0
failed = []
generated = 0

# Append to existing JSONL
with open(JSONL_PATH, "a") as manifest_file:
    for i, text in enumerate(TEXTS):
        clip_num = START_IDX + i + 1
        label = f"clip_{clip_num:04d}"
        print(f"[{i+1}/{len(TEXTS)}] (#{clip_num}) {text[:55]}...")

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
            full_audio = np.concatenate([full_audio, silence_1s])
            duration = len(full_audio) / 24000
            total_ms = (time.time() - t0) * 1000

            out_path = os.path.join(OUTPUT_DIR, "audio", f"{label}.wav")
            sf.write(out_path, full_audio, 24000)

            entry = {"audio": f"./audio/{label}.wav", "text": text, "ref_audio": "./ref.wav"}
            manifest_file.write(json.dumps(entry) + "\n")
            manifest_file.flush()

            generated += 1
            total_audio_duration += duration
            print(f"  {total_ms:.0f}ms | {duration:.1f}s | Batch total: {total_audio_duration/60:.1f}min")

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((clip_num, text, str(e)))

print(f"\n{'='*60}")
print(f"Batch 2 done! Generated {generated} clips")
print(f"Batch 2 audio: {total_audio_duration/60:.1f} minutes")
print(f"Failed: {len(failed)}")
print(f"Total clips in manifest: {START_IDX + generated}")
total_lines = sum(1 for _ in open(JSONL_PATH))
print(f"JSONL lines: {total_lines}")
if failed:
    print(f"\nFailed clips:")
    for idx, txt, err in failed:
        print(f"  [{idx}] {txt[:50]}... — {err}")
