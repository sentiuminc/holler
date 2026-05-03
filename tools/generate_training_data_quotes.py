"""Generate training data v2 — inspirational quotes + emotional variety.

Same generation engine as v1, but replaces generic factual statements with
curated quotes from Elon Musk, Steve Jobs, Hunter S. Thompson, Richard Hamming,
David Foster Wallace, and other thinkers from the Antidote to Slop collection.

Emotional/functional/assistant categories kept from v1.

Usage:
  python generate_training_data_v2.py --voice nora --ref-text "Oh wow, that actually worked!..."
  python generate_training_data_v2.py --voice joe --ref-text "..." --start 200
"""
import argparse
import time
import os
import sys
import json
import numpy as np
import soundfile as sf
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

VOICES_DIR = Path(__file__).parent.parent / "voices"

TEXTS = [
    # =========================================================================
    # KEPT FROM V1: Functional / emotional / assistant categories
    # =========================================================================

    # === Short responses (2-4 words) ===
    "Got it.", "Sure thing.", "Absolutely.", "Not quite.", "Let me check.",
    "One moment.", "That works.", "Good idea.", "Interesting.", "I see.",

    # === Short sentences (5-10 words) ===
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

    # === Questions ===
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

    # === More questions — curious / probing ===
    "What if we tried it the other way around?",
    "Have you thought about what happens if this fails?",
    "What's the worst case scenario here?",
    "Is this something we need right now, or is it more of a nice to have?",
    "What would you do if you had unlimited time on this?",
    "Who else should we loop in on this?",
    "Is there something I'm missing here?",
    "What's your gut telling you?",
    "Would it help if I showed you an example?",
    "Do you trust this data?",
    "How long have you been dealing with this?",
    "What would make this feel done to you?",
    "Is this a one-time thing or is it going to keep coming up?",
    "What's the simplest version of this that would actually work?",
    "If we could only fix one thing, which one would it be?",

    # === Emotional: enthusiastic ===
    "Oh, that's awesome! I didn't expect it to work this well on the first try.",
    "This is really exciting. I think we're onto something here.",
    "I love that idea. Let's absolutely do that.",
    "Wow, the results are way better than I thought they'd be!",
    "Yes! That's exactly the kind of thing I was hoping for.",
    "Wait, seriously? That's incredible!",
    "Oh man, this is so much better than what we had before!",
    "I can't believe that actually worked! Let's keep going!",
    "Dude, look at this. This is exactly what we needed!",
    "Hell yes! Okay, now we're cooking.",
    "That is genuinely impressive. Like, wow.",
    "I've been waiting for this to come together and it finally did!",
    "This might be the best thing we've built so far. Seriously.",
    "Oh nice! That's clean. Really, really clean.",
    "Okay now I'm excited. This changes the whole game.",

    # === Emotional: frustrated / determined ===
    "Okay, that didn't work. But I have another idea.",
    "Ugh, that's annoying. Let me try something different.",
    "Why is this breaking? It was working five minutes ago!",
    "Alright, third time's the charm. Let's go again.",
    "This shouldn't be this hard. Something's off.",
    "Come on. We're so close. Just one more thing.",
    "I'm not giving up on this. There has to be a way.",
    "That's frustrating, but at least we know what doesn't work.",
    "Okay, deep breath. Let's start from scratch on this part.",
    "No, that's not right either. But we're getting warmer.",

    # === Emotional: thoughtful/careful ===
    "Hmm, that's an interesting edge case. Let me think about how to handle it properly.",
    "I want to be careful here because any changes we make could affect other parts of the system.",
    "Let me take a step back and think about whether this is really the right approach.",
    "I'm not entirely sure about this one. There are some risks we should consider first.",
    "This requires a bit of nuance. The answer isn't as straightforward as it might seem.",

    # === Emotional: empathetic ===
    "I completely understand your frustration. Let's see what we can do to fix this.",
    "No worries at all, that's a totally reasonable question.",
    "I hear you. That does sound like a really annoying issue to deal with.",
    "Take your time, there's no rush. I'll be here whenever you're ready.",
    "I get it. Sometimes these things just don't work the way they should.",

    # === Emotional: matter-of-fact ===
    "The build succeeded. No errors or warnings.",
    "Total cost for the month was twelve dollars and forty seven cents.",
    "The server has been running for seventy two hours without any issues.",
    "Version three point five was released on Tuesday.",
    "Memory usage is currently at sixty eight percent.",

    # === Conversational filler / natural speech ===
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

    # === Acknowledgments / transitions ===
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

    # === Casual conversation and reactions ===
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

    # === Short and punchy ===
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

    # === Real ivi assistant lines ===
    "Doing good, Chris. What's up?",
    "Hey! All good here. What's up?",
    "Doing well. What's up?",
    "Anytime. Holler if you need anything.",
    "Anytime. Hit me up whenever you need something.",
    "Cool. What's the priority next?",
    "Got it. Ready whenever you are.",
    "Done. Monitor's at fifty percent.",
    "Done. Set to fifty percent.",
    "Done. Volume set to forty-two.",
    "Done, brightness is at fifty percent.",
    "Nothing's playing right now.",
    "Cool. What'd you find?",
    "Cool. What's on your mind?",
    "Didn't catch that, what do you want me to do?",
    "Go ahead, I'll check the logs in a moment.",
    "Get some rest when you can. Want me to put on something chill or dim the screen?",
    "All good. Hit me up anytime.",
    "Fair point. I guess I am, kinda. But hopefully way less annoying.",
    "Paused at fourteen seconds.",
    "What's wrong? The day, the weather, or both?",
    "What part? The date, the weather, or something else?",
    "Ha, so there's room to go faster? That's exciting. What are you optimizing next?",
    "That means a lot. I think there's something real here that works.",
    "Episode five is loading now.",
    "Confirmed. One day, eleven hours.",
    "Fair enough. What do you actually need from a browser?",
    "Glad it's working well. The curve is dialed in then.",
    "Cool, so it's moving in the right direction. What feels off?",
    "Is that a common one or just a one-off?",
    "Don't have that in my memory. Want me to check your history?",
    "About what? Something specific, or just thinking out loud?",
    "Depends if you want something cerebral or just want to chill with a good movie.",
    "I can control your Mac, brightness, volume, media playback. Search the web, read files, find stuff. Basically I'm here to help you get things done.",
    "A car engine uses controlled explosions to create power. Fuel and air mix in cylinders, get compressed, then ignite. Pretty straightforward once you break it down.",
    "That's a smart interface that learns from how you use it and adapts. It anticipates what you need and gives suggestions based on your behavior.",
    "It's Monday. Pretty windy right now, about twenty-eight miles per hour with gusts up to thirty-nine. Mostly sunny though.",
    "About a hundred fifteen to a hundred sixteen bucks, depending on the exchange rate.",
    "Enjoy. Let me know if you end up at one of those spots.",
    "Good. So start with the bank statement.",
    "Fullscreen activated. It should be playing on your main display now.",
    "Let me know if you want me to dig deeper on that.",
    "Hundred grams of raw carrot's got about two point eight grams of fiber. Pretty decent.",
    "Chia seeds are packed with fiber, protein, and omega-three fatty acids. Pretty dense nutrition for a tiny seed.",
    "The best scrambled eggs are cooked low and slow. High heat makes them rubbery. Trust me on this one.",
    "Clippy was ahead of its time, honestly. The idea was solid. Just terrible execution.",
    "About four forty in the morning. Still night over there.",
    "Eight thirty-seven PM, Tuesday night.",
    "Converting that for you now. One second.",
    "Your name's Chris. You've got a personal AI setup going.",
    "Heard. You want me to crank up the responsiveness, or add something on top?",
    "Fair point. The data I pulled was a bit stale. Let me check again.",

    # === Excited / high energy ===
    "Oh wow, that actually worked on the first try! I was not expecting that.",
    "This is amazing. Look at these numbers, way better than what we had before!",
    "Yes! That's exactly what we needed. Let's keep going.",
    "Okay I'm genuinely impressed. This is really, really good.",
    "No way. That's actually incredible. How did you figure that out?",
    "The results just came in and honestly? They blew my expectations out of the water.",
    "I'm so excited about this. Seriously, this changes everything.",
    "We did it! Everything's green across the board.",
    "That's a breakthrough. Like, an actual breakthrough. Not the Silicon Valley kind.",
    "Holy cow, the latency dropped to under a hundred milliseconds. That's insane.",

    # === Surprised / reactive ===
    "Wait, what? Say that again.",
    "No way! Are you serious right now?",
    "Hold on, that changes everything!",
    "Okay, I did not see that coming.",
    "Shut up. That's actually real?",
    "Oh! Oh, I see what you did there.",
    "Whoa. That's way more than I expected.",
    "Huh! That's actually kind of genius.",
    "Get out of here. That worked?",
    "Oh snap, we were doing it wrong the whole time!",

    # === Encouraging / pumping up ===
    "You've got this. Seriously.",
    "Trust yourself on this one. Your instincts are good.",
    "Hey, you're further along than you think you are.",
    "Don't second-guess it. That was the right call.",
    "You're building something real here. Keep going.",
    "The hard part is already done. Everything else is details.",
    "Look how far you've come since you started.",
    "That took guts. I respect that.",
    "You made the right choice. I'm sure of it.",
    "Keep pushing. You're almost there.",

    # === Calm / slow / thoughtful ===
    "Take your time. There's no rush at all. I'll be here whenever you're ready.",
    "Let me think about this for a moment. I want to make sure I give you a good answer.",
    "Hmm. That's a really interesting question. Let me sit with it for a second.",
    "You know, sometimes the best thing to do is nothing. Just let it settle.",
    "I think... yeah, I think you might be right about that. It's worth considering.",
    "There's no wrong answer here. It really comes down to what feels right to you.",
    "I hear you. That's a tough one. Let's think through it together.",
    "Sleep on it. Things usually look clearer in the morning.",
    "No pressure. Whenever you're ready, I'm here.",
    "That's a big decision. Don't rush it.",

    # === Ultra-short ===
    "Yep.", "Nope.", "Done.", "On it.", "Sure.", "Okay.", "Got it.", "Nice.",
    "Makes sense.", "Fair enough.", "Good call.", "Smart move.", "Let's go.",
    "Totally.", "Exactly.", "Agreed.", "Perfect.", "Right.",
    "Not quite.", "Almost.", "Close enough.", "That works.", "Even better.",

    # === Empathetic / warm ===
    "Hey, rough day? Want to talk about it, or just need me to handle some stuff quietly?",
    "I can tell this one's been frustrating. Let me see if I can help.",
    "You've been at this for a while. Maybe take a quick break? I'll hold down the fort.",
    "That sounds really stressful. What can I do to make it easier?",
    "I'm sorry to hear that. Is there anything I can help with right now?",
    "You're doing great, by the way. Even if it doesn't feel like it.",
    "Sometimes things just don't go the way you plan. That's okay.",
    "I appreciate you trusting me with this.",
    "No judgment here. We'll figure it out together.",
    "Take a breath. We've got time.",

    # === Informational / reading out loud ===
    "Next up: your three o'clock call with the design team. They want to review the latest mockups.",
    "The weather tomorrow looks great. Sunny, twenty-four degrees, light breeze from the west.",
    "You've got four unread messages. Two from your team, one from the bank, and one from your mom.",
    "Your flight lands at six fifteen local time. That gives you about two hours to get to the hotel.",
    "The package shipped yesterday and should arrive by Thursday.",
    "Battery's at thirty-eight percent. You might want to plug in before the call.",
    "The meeting got moved to four thirty. I've updated your calendar.",
    "Traffic looks clear right now. Should take about twenty-five minutes to get there.",
    "The restaurant closes at ten, so you've got about an hour and a half.",
    "Your subscription renews in three days. Just a heads up.",

    # =========================================================================
    # NEW IN V2: Curated quotes — Elon, Jobs, Thompson, Hamming, DFW, others
    # =========================================================================

    # === Elon Musk — mindset & purpose ===
    "You can choose to be not ordinary.",
    "It's possible for ordinary people to choose to be extraordinary.",
    "Don't aspire to glory. Aspire to work.",
    "A useful life is worth having lived.",
    "Fight for the things that make you excited about the future.",
    "The future will not get here fast enough unless we force it.",
    "I'm motivated by curiosity more than anything.",
    "This is the foundation of my philosophy: I am curious about the nature of the universe.",
    "Remember the future.",
    "Doing something I enjoy, which is useful for other people, that gives me satisfaction.",
    "If heat death will inevitably end the universe, it actually is all about the journey.",
    "Life is too short to spend it doing something you don't like.",
    "If you need encouragement, don't start a company.",
    "Nobody ever changed the world on forty hours a week.",

    # === Elon Musk — fear & perseverance ===
    "Look fear straight in the eye and it will disappear.",
    "It's normal to feel fear. Just feel it and let the importance of your mission drive you to do it anyway.",
    "We should not be afraid of doing something just because some amount of tragedy is likely to occur.",
    "I don't ever give up. I'd have to be dead or completely incapacitated.",
    "Adversity shaped me. My pain threshold became very high.",
    "You have to feel quite compelled to start a company. You must have a high pain threshold.",

    # === Elon Musk — truth & first principles ===
    "I try to be hyperrational.",
    "I am obsessed with truth. If you're going to come up with a good solution, the truth is really, really important.",
    "Physics is law. Everything else is a recommendation.",
    "Truth matters to me a lot. Pathologically, it matters to me.",
    "It's OK to be wrong. Just don't be confident and wrong.",
    "Don't just follow the trends. Think with the physics approach, first principles. It's a powerful method for life in general.",
    "It's hard to think this way. It takes a lot of effort. But if you're trying to do something new, it's the best way to think.",
    "The most common mistake of smart engineers is to optimize a thing that should not exist.",
    "Impossible is a strong word. I approach things from a physics standpoint, and impossible is more or less banned.",
    "The best part is no part. The best process is no process.",
    "Simplicity creates both reliability and low cost.",

    # === Elon Musk — learning & people ===
    "Most people can learn a lot more than they think they can. They sell themselves short by not trying.",
    "I encourage you to read a lot of books. Just read. Try to ingest as much information as you can.",
    "It is important to view knowledge as a semantic tree. Make sure you understand the fundamental principles before you get into the details.",
    "Most people self-limit their ability to learn. It's pretty straightforward: just read books and talk to people.",
    "Talk to people from different walks of life, in different industries and professions. Try to learn as much as possible.",
    "The most important thing is to attract great people.",
    "A company is just a bunch of people coming together to create a product or service.",
    "Wherever the smartest, most driven people are choosing to work, that company is going to win.",
    "When hiring, look for people with the right attitude. Skills can be taught. Attitude changes require a brain transplant.",
    "A small group of technically strong people will always beat a large group of moderately strong people.",

    # === Elon Musk — execution & urgency ===
    "No matter how smart you are, you will make some number of mistakes. Everyone makes mistakes.",
    "In business and personal life, wishful thinking causes a lot of mistakes. You have to ask whether something is true or not.",
    "Being tenacious and super focused on the truth is extremely important. Look for feedback from all sources.",
    "All bad news should be given loudly and often. Good news can be said quietly and once.",
    "Never ask your troops to do something you're not willing to do.",
    "Walk out of a meeting as soon as it is obvious you aren't adding value. It is not rude to leave. It is rude to waste someone's time.",
    "The one thing you cannot replace is time.",
    "If a timeline is long, it's wrong.",
    "A maniacal sense of urgency is our operating principle.",
    "Pay close attention to negative feedback, and solicit it, particularly from friends. It's incredibly helpful.",
    "Better to pick a path and keep moving than just vacillate endlessly on a decision.",
    "Life is too short for long-term grudges.",

    # === Elon Musk — building & entrepreneurship ===
    "What is a useful thing you could build that you wish existed in the world?",
    "Try to find an overlap of your talents and what you're interested in. You may have skill in something but don't like doing it.",
    "If you're creating something you love and think other people will love, it's much easier to sacrifice the time and effort.",
    "When starting a company, create a demonstration, a mock-up, or a sketch. Try to get to that point as soon as possible.",
    "Fundamentally, if you don't have a compelling product at a compelling price, you don't have a great company.",
    "Prototypes are easy and fun. Reaching volume production with a reliable product at an affordable price is excruciatingly difficult.",
    "The biggest epiphany I had building Tesla is what really matters is the machine that builds the machines.",
    "When you have a big technology change, it tends to come from new companies.",
    "If conventional thinking makes your mission impossible, then unconventional thinking is necessary.",
    "Technological progress is not inevitable. Humans make technology. If we don't do it, it will not happen.",

    # === Elon Musk — vision & civilization ===
    "I can't emphasize this enough: As long as we push hard and are not complacent, the future is going to be great.",
    "If you don't push for radical breakthroughs, you're not going to get radical outcomes.",
    "If you want the future to be good, you must make it so.",
    "When something is important enough, you do it even if the odds are not in your favor.",
    "There must be things to inspire us, that make you proud to be a member of humanity.",
    "It is time to go forth, be out there among the stars. Expand the scope and scale of human consciousness.",
    "This is the first moment in four-and-a-half billion years that it has been possible to extend life beyond Earth.",
    "We should view our civilization as much more fragile than we think.",
    "Things don't always go up. Look at the history of civilizations: they rise and they fall.",
    "Go do it. Just go out there and do it. People are far too afraid to try. Fear is the biggest reason for failure.",
    "A major failure mode is a high ego-to-ability ratio. If it gets too high, you've broken the feedback loop to reality.",
    "Always be smashing your ego. Internalize responsibility. Do whatever it takes to succeed.",

    # === Steve Jobs — craft & products ===
    "The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle.",
    "Design is not just what it looks like and feels like. Design is how it works.",
    "Simple can be harder than complex. You have to work hard to get your thinking clean enough to make it simple.",
    "Innovation distinguishes between a leader and a follower.",
    "Stay hungry. Stay foolish.",
    "You can't connect the dots looking forward. You can only connect them looking backwards.",
    "Have the courage to follow your heart and intuition. They somehow already know what you truly want to become.",
    "Quality is more important than quantity. One home run is much better than two doubles.",

    # === Steve Jobs — thinking & business ===
    "A lot of things in business are what I call folklore. They're done because they were done yesterday and the day before.",
    "If you're willing to ask a lot of questions and think about things and work really hard, you can learn business pretty fast.",
    "Throughout the years in business, I found something. Nobody knows why they do what they do. Nobody thinks about things very deeply.",
    "People get confused that the process is the content. The best people are the ones who really understand the content.",
    "It was very clear to me that for every hardware hobbyist, there were a thousand people who just wanted to mess around with programming.",
    "We built these things for ourselves because we couldn't afford to buy anything.",
    "What we learned was that we could build something ourselves that could control billions of dollars' worth of infrastructure. That was an incredible lesson.",
    "I think everybody should learn how to program a computer. It teaches you how to think.",
    "I view computer science as a liberal art.",

    # === Steve Jobs — teams & excellence ===
    "A team of people doing something they really believe in is like a rock tumbler. They polish each other and they polish the ideas.",
    "There's a tremendous amount of craftsmanship between a great idea and a great product.",
    "Designing a product is keeping five thousand things in your brain and fitting them all together in new and different ways.",
    "The heaviness of being successful was replaced by the lightness of being a beginner again.",
    "Your work is going to fill a large part of your life, and the only way to be truly satisfied is to do what you believe is great work.",
    "Sometimes life's going to hit you in the head with a brick. Don't lose faith.",
    "Your time is limited, so don't waste it living someone else's life.",
    "Don't be trapped by dogma, which is living with the results of other people's thinking.",
    "Don't let the noise of others' opinions drown out your own inner voice.",
    "It was worth about over a million dollars when I was twenty-three. And it wasn't that important, because I never did it for the money.",

    # === Steve Jobs — vision ===
    "When we saw the graphical user interface, within ten minutes it was obvious that all computers would work like this someday.",
    "The product sensibility that brought them to that monopolistic position gets rotted out by people who have no conception of a good product.",
    "What is simple in one arena is often profound in another.",
    "The people at the research center used to call the people that ran the company toner heads. They had no clue about what they were seeing.",

    # === Hunter S. Thompson — purpose & choosing ===
    "Whether to float with the tide, or to swim for a goal. It is a choice we must all make at one time in our lives.",
    "Every man is the sum total of his reactions to experience. As your experiences differ, you become a different man.",
    "We strive to be ourselves. That is the real goal.",
    "A man must choose a path which will let his abilities function at maximum efficiency toward the gratification of his desires.",
    "He has not dedicated his life to reaching a pre-defined goal, but he has rather chosen a way of life he knows he will enjoy.",
    "The goal is absolutely secondary. It is the functioning toward the goal which is important.",
    "To let another man define your own goals is to give up one of the most meaningful aspects of life.",
    "If you can't see any real purpose in any of the paths, you must find a new one.",
    "A man who procrastinates in his choosing will inevitably have his choice made for him by circumstance.",
    "Beware of looking for goals. Look for a way of life. Decide how you want to live and then see what you can do to make a living within that way of life.",
    "It is not necessary to accept the choices handed down to you by life as you know it. There is more to it than that.",
    "Make the goal conform to the individual, rather than make the individual conform to the goal.",

    # === Richard Hamming — doing great work ===
    "You have one life to lead. You might as well lead a life you would like to have.",
    "I suggest you a life of doing something significant. By your definition of significant.",
    "If you don't work on important problems, you are not going to do important things except by the dumbest of dumb luck.",
    "If what you're working on is not important and it's not likely to lead to important things, why are you working on it?",
    "Luck favors a prepared mind. You prepare yourself day to day. When the lightning strikes, you're either ready or you're not.",
    "To a great extent, it is constant hard work that does it. Nothing more and nothing less.",
    "The most important thing of great people is they believe they can do great work. If you don't think you can, it's not likely you ever will.",
    "You should study your successes. Because when your time comes, you will know how to succeed. If you study failures, you'll know how to fail.",
    "When you have a vision, you will go a long way. Without a vision of what you're going to do and where you're going to be, you're not going to get very far.",
    "Whatever you do, you're going to do well. Excellence is one of the best tracks you can use.",
    "What appeared to be a defect, by turning the problem around, became an asset.",
    "Frequently, when you think things are wrong and you haven't got the wherewithal, if you turn the problem around, you can turn it into a great success.",
    "Those who work with the door shut may be working just as hard ten years later, but they don't know what to work on. They are not connected with reality.",
    "I ain't scared of nothing. Let's go ahead and see what happens.",
    "The difference between being strong willed and stubborn, and the difference between confidence and overconfidence, is about the same thing. It's this fine line.",

    # === David Foster Wallace — attention & awareness ===
    "The most obvious, important realities are often the ones that are hardest to see and talk about.",
    "The really significant education in thinking is not about the capacity to think, but rather about the choice of what to think about.",
    "Learning how to think really means learning how to exercise some control over how and what you think.",
    "If you cannot exercise this kind of choice in adult life, you will be totally hosed.",
    "A huge percentage of the stuff that I tend to be automatically certain of is, it turns out, totally wrong and deluded.",
    "Everything in my own immediate experience supports my deep belief that I am the absolute center of the universe. We rarely think about this because it's so socially repulsive.",
    "The really important kind of freedom involves attention and awareness and discipline, and being able truly to care about other people.",
    "That is real freedom. That is being educated, and understanding how to think.",
    "You get to consciously decide what has meaning and what doesn't. You get to decide what to worship.",
    "Everybody worships. The only choice we get is what to worship.",
    "If you worship money and things, you will never have enough, never feel you have enough.",
    "The whole trick is keeping the truth up front in daily consciousness.",
    "It is unimaginably hard to stay conscious and alive in the adult world day in and day out.",

    # === Freedom & money — JL Collins ===
    "There are many things money can buy, but the most valuable of all is freedom. Freedom to do what you want and work for whom you respect.",
    "Those who carry debt are slaves with even stouter shackles. Don't think for a moment their masters don't know it.",
    "I may never own a Mercedes but I'll always be able to say what needs to be said when it needs to be said.",
    "Keep your personal burn rate low. This alone will give you a lot of opportunities in life.",
    "Whether or not money can buy happiness, it can buy freedom, and that's a big deal.",
    "Making money is often more fun than spending it.",

    # === Wisdom — mixed sources from the Antidote to Slop ===
    "Life is not a dress rehearsal. This is probably it. Make it count.",
    "Don't do stuff that doesn't make you happy. This happens most often when other people want you to do something.",
    "Don't chase status. Status without substance doesn't work for long and is unfulfilling.",
    "Be a doer, not a talker.",
    "Forgive people.",
    "Think for a few seconds before you act. Think for a few minutes if you're angry.",
    "Don't judge other people too quickly. You never know their whole story.",
    "Don't worry so much. Things in life are rarely as risky as they seem. Most people are too risk-averse.",
    "Ask for what you want.",
    "If you think you're going to regret not doing something, you should probably do it.",
    "Exercise. Eat well. Sleep. Get out into nature with some regularity.",
    "Go out of your way to help people. Few things in life are as satisfying.",
    "Learn voraciously.",
    "Do new things often. Not only does it slow down the perception of time, it increases happiness and keeps life interesting.",
    "Don't screw people and don't burn bridges. Pick your battles carefully.",
    "Most things are ok in moderation. Almost nothing is ok in extreme amounts.",
    "Given enough time, it is possible to adjust to almost anything, good or bad. Humans are remarkable at this.",
    "Go out of your way to be around smart, interesting, ambitious people. It really is true that you become an average of the people you spend the most time with.",
    "Minimize your cognitive load from distracting things that don't really matter.",
    "The days are long, but the decades are short.",

    # === Craft, creativity & work ===
    "Creativity isn't about having original ideas. It's about making unexpected connections between existing ones.",
    "The difference between knowledge and wisdom is that knowledge is knowing what to do, and wisdom is knowing when to do it.",
    "Simple doesn't mean easy. Making something simple often requires more effort than making something complex.",
    "Perfection is the enemy of done. At some point you have to ship it and iterate.",
    "The best way to learn something is to try to teach it to someone else. It exposes all the gaps in your understanding.",
    "Confidence isn't about knowing you're right. It's about being comfortable with being wrong.",
    "Asking the right question is often more important than having the right answer.",
    "Progress isn't always visible. Some of the most important changes happen slowly, beneath the surface.",
    "The stories we tell ourselves about who we are have a bigger impact on our behavior than objective reality does.",
    "Mistakes are just data points. They tell you what doesn't work, which is genuinely valuable information.",
    "The map is not the territory. Our models of reality are always simpler than reality itself.",
    "The best designs are invisible. You only notice design when it's bad.",
    "Multitasking is a myth for most cognitive tasks. Your brain actually switches between tasks, and each switch costs you time.",
    "Deep work requires at least twenty to thirty minutes of uninterrupted focus to get into a flow state.",
    "Writing things down helps you remember them, even if you never look at the notes again. The act of writing itself is what matters.",

    # === Relationships & human nature ===
    "Listening is the most underrated skill in communication. Most people are just waiting for their turn to speak.",
    "Small acts of kindness compound over time. They might seem insignificant in the moment, but they add up.",
    "People remember how you made them feel long after they forget what you said.",
    "Trust is built in drops and lost in buckets. It takes years to establish and seconds to destroy.",
    "Everyone you meet knows something you don't. That's a good reason to stay curious about other people.",
    "The people who have the biggest impact on your life often don't realize it.",
    "The best friendships are the ones where you can pick up right where you left off, even after months of not talking.",
    "The hardest part of any disagreement is genuinely trying to understand the other person's perspective.",
    "Saying no is a skill that most people struggle with, but it's essential for protecting your time and energy.",
    "People tend to overestimate what they can do in a day and underestimate what they can do in a year.",

    # === Timeless philosophy ===
    "The only constant is change. That's been true for thousands of years, which is kind of ironic.",
    "We spend so much time planning for the future that we forget to pay attention to what's happening right now.",
    "Most of the things people worry about never actually happen. But telling someone not to worry doesn't help.",
    "Sometimes the hardest part of solving a problem is admitting that it exists in the first place.",
    "At the end of the day, what matters most is that it works reliably.",
    "If I'm being honest, I wasn't sure this would work at first.",
    "Looking at the bigger picture, I think we're in really good shape.",
    "Speaking from experience, these kinds of issues usually have a simple fix.",
    "For what it's worth, I think your instinct on this was correct.",
    "Long story short, we need to update the configuration and redeploy.",

    # === Music, art & beauty ===
    "Music has this incredible ability to transport you back to a specific moment in time. Just a few notes can trigger a flood of memories.",
    "Jazz is about the notes you don't play as much as the ones you do. The silence between phrases is where the magic happens.",
    "Photography changed how humans see the world. For the first time, you could freeze a moment exactly as it happened.",
    "There's a theory that all stories follow one of about seven basic plot structures. Everything else is variation.",
    "Every font tells a story before you even read the words. Typography is one of those things most people never consciously think about.",
    "Abstract art isn't about what it looks like. It's about what it makes you feel.",
    "A good book can change how you think about the world. That's a lot of power for a few hundred pages.",
    "Short stories are an underrated art form. Telling a complete story in a few pages requires incredible skill.",

    # === Observations on life ===
    "There's a specific kind of tiredness that comes from being productive all day. It feels different from doing nothing.",
    "Sometimes the best part of the day is that first cup of coffee in the morning, when everything is still quiet.",
    "People underestimate how much a good night's sleep can change your perspective on a problem.",
    "Walking is underrated as exercise. A thirty minute walk every day can do wonders for both physical and mental health.",
    "The best conversations usually happen when you're not trying to have a great conversation. They just flow naturally.",
    "A clean desk doesn't necessarily mean a productive person, but it does tend to reduce mental clutter.",
    "The best travel advice is to pack half of what you think you need and bring twice the money.",
    "Learning even ten words of the local language can completely change how people treat you when you travel.",
    "The best souvenirs aren't objects. They're stories and memories that you carry with you.",
    "The difference between traveling and tourism is how much you're willing to get lost.",

    # === Open source & technology ===
    "Open source software powers most of the internet. The biggest projects are maintained by volunteers.",
    "The internet was originally designed to survive a nuclear war. Now we use it to share pictures of our lunch.",
    "Email was supposed to make communication faster. Instead, it created an entirely new category of stress.",
    "Cloud computing basically means using someone else's computer. The metaphor makes it sound more magical than it is.",
    "The first website ever created is still online. It's a plain text page about the World Wide Web project.",

    # === Multi-sentence wisdom ===
    "Here's my suggestion. Let's keep the current version running for now and work on the update in parallel. That way we don't risk any downtime.",
    "So there's good news and not so good news. The good news is the core functionality works perfectly. The not so good news is we need to redo the interface.",
    "Okay, so I looked into the issue. It turns out the problem was on our side, not theirs. Easy fix once you know where to look.",
    "The way I see it, we have two main options. We can either fix the existing system or build something new from scratch. Each has its own set of tradeoffs.",
    "What I'd suggest is that we start with the simplest possible version, get that working reliably, and then gradually add the more complex features on top.",
]


BENCHMARK_TEXTS = TEXTS[:20]


def run_benchmark(voice_dir, ref_text):
    """Run 20 clips and report per-clip timing + summary."""
    ref_audio = str(voice_dir / "ref.wav")
    if not Path(ref_audio).exists():
        print(f"ERROR: {ref_audio} not found.")
        return

    print("Loading Qwen3-TTS 1.7B Base bf16...")
    t0 = time.time()
    from mlx_audio.tts import load
    model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Warmup...")
    for _ in model.generate(text="Hello.", ref_audio=ref_audio, ref_text=ref_text, language="english", temperature=0.85):
        pass
    print("Warmup done.\n")

    timings = []
    durations = []
    total_start = time.time()

    for i, text in enumerate(BENCHMARK_TEXTS):
        print(f"[{i+1}/20] {text[:60]}...")
        t0 = time.time()
        chunks = []
        for result in model.generate(
            text=text, ref_audio=ref_audio, ref_text=ref_text,
            language="english", temperature=0.85,
        ):
            audio = np.array(result.audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            chunks.append(audio)

        full_audio = np.concatenate(chunks)
        elapsed_ms = (time.time() - t0) * 1000
        dur = len(full_audio) / 24000
        rtf = elapsed_ms / 1000 / dur if dur > 0 else 0
        timings.append(elapsed_ms)
        durations.append(dur)
        print(f"  {elapsed_ms:.0f}ms | {dur:.1f}s audio | RTF {rtf:.2f}")

    total_elapsed = time.time() - total_start
    total_audio = sum(durations)
    overall_rtf = total_elapsed / total_audio if total_audio > 0 else 0

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS (20 clips)")
    print(f"{'='*60}")
    print(f"Total wall time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Total audio:      {total_audio:.1f}s ({total_audio/60:.1f}min)")
    print(f"Overall RTF:      {overall_rtf:.2f}")
    print(f"Avg per clip:     {sum(timings)/len(timings):.0f}ms")
    print(f"Min per clip:     {min(timings):.0f}ms")
    print(f"Max per clip:     {max(timings):.0f}ms")
    print(f"Projected 500:    {total_elapsed/20*500/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser(description="Generate training data v2 (inspirational texts)")
    parser.add_argument("--voice", required=True, help="Voice name (directory under voices/)")
    parser.add_argument("--ref-text", required=True, help="Transcript of ref.wav")
    parser.add_argument("--start", type=int, default=0, help="Resume from this clip index (0-based)")
    parser.add_argument("--benchmark", action="store_true", help="Run 20-clip benchmark only, no file output")
    args = parser.parse_args()

    voice_dir = VOICES_DIR / args.voice

    if args.benchmark:
        run_benchmark(voice_dir, args.ref_text)
        return

    training_dir = voice_dir / "training-data"
    ref_audio = str(voice_dir / "ref.wav")
    out_dir = training_dir / "audio-original"
    jsonl_path = training_dir / "train.jsonl"

    if not Path(ref_audio).exists():
        print(f"ERROR: {ref_audio} not found. Place ref.wav in voices/{args.voice}/")
        return

    os.makedirs(out_dir, exist_ok=True)

    texts = TEXTS[args.start:]
    total_texts = len(TEXTS)
    print(f"Voice: {args.voice}")
    print(f"Ref: {ref_audio}")
    print(f"Ref text: {args.ref_text[:80]}...")
    print(f"Output: {out_dir}")
    print(f"Texts: {len(texts)} (starting at {args.start}, total corpus {total_texts})")
    print()

    print("Loading Qwen3-TTS 1.7B Base bf16...")
    t0 = time.time()
    from mlx_audio.tts import load
    model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Warmup...")
    for _ in model.generate(text="Hello.", ref_audio=ref_audio, ref_text=args.ref_text, language="english", temperature=0.85):
        pass
    print("Warmup done.\n")

    existing_count = 0
    if args.start > 0 and jsonl_path.exists():
        with open(jsonl_path) as f:
            existing_count = sum(1 for _ in f)
        print(f"Existing manifest has {existing_count} entries. Appending new clips.\n")

    total_audio_duration = 0
    failed = []
    generated = 0

    jsonl_mode = "a" if args.start > 0 else "w"
    with open(jsonl_path, jsonl_mode) as manifest_file:
        for i, text in enumerate(texts):
            clip_idx = args.start + i
            label = f"clip_{clip_idx+1:04d}"
            print(f"[{clip_idx+1}/{total_texts}] {text[:60]}...")

            t0 = time.time()
            chunks = []

            try:
                for result in model.generate(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=args.ref_text,
                    language="english",
                    temperature=0.85,
                ):
                    audio = np.array(result.audio, dtype=np.float32)
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                    chunks.append(audio)

                full_audio = np.concatenate(chunks)
                duration = len(full_audio) / 24000
                total_ms = (time.time() - t0) * 1000

                out_path = out_dir / f"{label}.wav"
                sf.write(str(out_path), full_audio, 24000)

                entry = {
                    "audio": f"./audio/{label}.wav",
                    "text": text,
                    "ref_audio": "./ref.wav",
                    "ref_text": args.ref_text,
                }
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()

                generated += 1
                total_audio_duration += duration
                print(f"  {total_ms:.0f}ms | {duration:.1f}s | Total: {total_audio_duration/60:.1f}min")

            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append((clip_idx, text, str(e)))

    print(f"\n{'='*60}")
    print(f"Done! Generated {generated}/{len(texts)} clips")
    print(f"Total audio: {total_audio_duration/60:.1f} minutes")
    print(f"Total manifest entries: {existing_count + generated}")
    print(f"Failed: {len(failed)}")
    print(f"Output: {out_dir}")
    print(f"Manifest: {jsonl_path}")
    if failed:
        print(f"\nFailed clips:")
        for idx, txt, err in failed:
            print(f"  [{idx}] {txt[:50]}... -- {err}")


if __name__ == "__main__":
    main()
