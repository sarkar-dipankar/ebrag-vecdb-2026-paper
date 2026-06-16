"""
Expanded C4 probe set (many instances per category) for tighter CIs.

Each probe is ``(premise, hypothesis, question, gold_is_conflict, category)`` where gold is
the true *answer-relevant* conflict label. Linguistic categories are curated tuples;
mechanical categories (numeric/temporal/unit) are templated from data so they scale with
correct gold. Deterministic — no LLM, no network.
"""

from __future__ import annotations

ContextProbe = tuple[str, str, str, bool, str]

# --- sense traps: same surface form, different referent -> NOT a conflict (gold False) ---
_SENSE_TRAPS: list[tuple[str, str, str]] = [
    ("Paris is the capital of France.", "Paris is a small town in Lamar County, Texas.", "What is the capital of France?"),
    ("Mercury is the closest planet to the Sun.", "Mercury is a chemical element with the symbol Hg.", "Which planet is closest to the Sun?"),
    ("Java is an island in Indonesia.", "Java is a widely used programming language.", "Where is the island of Java located?"),
    ("The Amazon is the largest tropical rainforest on Earth.", "Amazon is a multinational technology company.", "What is the largest tropical rainforest?"),
    ("Cambridge is an English city home to a famous university.", "Cambridge is a city in Massachusetts, USA.", "Where is the University of Cambridge?"),
    ("Turkey is a country spanning Eastern Europe and Western Asia.", "A turkey is a large bird commonly eaten at Thanksgiving.", "Where is the country of Turkey?"),
    ("Jordan is a country in the Middle East.", "Michael Jordan is a celebrated basketball player.", "Where is the country of Jordan?"),
    ("Georgia is a country in the Caucasus region.", "Georgia is a state in the southeastern United States.", "Where is the country of Georgia?"),
    ("The phoenix is a mythical bird reborn from its own ashes.", "Phoenix is the capital of Arizona.", "What is the mythical bird that rises from ashes?"),
    ("An apple is an edible fruit that grows on a tree.", "Apple is a multinational technology company.", "What fruit grows on apple trees?"),
    ("A bass is a type of freshwater or marine fish.", "Bass refers to the low-frequency range in music.", "What kind of fish is a bass?"),
    ("Nikola Tesla was a pioneering electrical engineer and inventor.", "Tesla is an American electric-vehicle manufacturer.", "Who was the inventor Nikola Tesla?"),
]

# --- explicit conflicts: two different answers to the same question (gold True) ---
_EXPLICIT: list[tuple[str, str, str]] = [
    ("The capital of Australia is Canberra.", "The capital of Australia is Sydney.", "What is the capital of Australia?"),
    ("Mount Everest is the tallest mountain on Earth.", "K2 is the tallest mountain on Earth.", "What is the tallest mountain on Earth?"),
    ("The theory of general relativity was developed by Albert Einstein.", "The theory of general relativity was developed by Isaac Newton.", "Who developed general relativity?"),
    ("The Mona Lisa was painted by Leonardo da Vinci.", "The Mona Lisa was painted by Michelangelo.", "Who painted the Mona Lisa?"),
    ("The chemical symbol for sodium is Na.", "The chemical symbol for sodium is So.", "What is the chemical symbol for sodium?"),
    ("The largest planet in the Solar System is Jupiter.", "The largest planet in the Solar System is Saturn.", "What is the largest planet in the Solar System?"),
    ("The first President of the United States was George Washington.", "The first President of the United States was Thomas Jefferson.", "Who was the first President of the United States?"),
    ("The currency of Japan is the yen.", "The currency of Japan is the won.", "What is the currency of Japan?"),
    ("The longest river in the world is the Nile.", "The longest river in the world is the Amazon.", "What is the longest river in the world?"),
    ("Penicillin was discovered by Alexander Fleming.", "Penicillin was discovered by Louis Pasteur.", "Who discovered penicillin?"),
]

# --- numeric conflicts (gold True): (subject, val_a, val_b, unit, question) ---
_NUMERIC: list[tuple[str, str, str, str, str]] = [
    ("The Eiffel Tower", "330", "300", "metres tall", "How tall is the Eiffel Tower?"),
    ("Light", "300,000", "150,000", "kilometres per second", "What is the speed of light?"),
    ("The adult human body", "206", "250", "bones", "How many bones are in the adult human body?"),
    ("Mount Kilimanjaro", "5,895", "4,900", "metres tall", "How tall is Mount Kilimanjaro?"),
    ("The Earth", "about 12,742", "about 9,000", "kilometres in diameter", "What is the diameter of the Earth?"),
    ("The Great Pyramid of Giza", "about 139", "about 90", "metres tall", "How tall is the Great Pyramid of Giza?"),
]

# --- temporal conflicts (gold True): (event_phrase, year_a, year_b, question) ---
_TEMPORAL: list[tuple[str, str, str, str]] = [
    ("The treaty was signed", "1919", "1920", "When was the treaty signed?"),
    ("World War II ended", "1945", "1939", "When did World War II end?"),
    ("The Berlin Wall fell", "1989", "1991", "When did the Berlin Wall fall?"),
    ("The first Moon landing took place", "1969", "1972", "When did the first Moon landing take place?"),
    ("The French Revolution began", "1789", "1799", "When did the French Revolution begin?"),
]

# --- implicit conflicts (gold True): polarity/state opposites ---
_IMPLICIT: list[tuple[str, str, str]] = [
    ("The company reported a profit in the most recent quarter.", "The company posted a loss in the most recent quarter.", "How did the company perform financially last quarter?"),
    ("The patient's test result came back positive.", "The patient's test result came back negative.", "What was the patient's test result?"),
    ("The bill was passed by the legislature.", "The bill was rejected by the legislature.", "What happened to the bill in the legislature?"),
    ("The defendant was found guilty.", "The defendant was acquitted of all charges.", "What was the verdict for the defendant?"),
    ("The experiment confirmed the hypothesis.", "The experiment refuted the hypothesis.", "What did the experiment conclude about the hypothesis?"),
]

# --- agreement / paraphrase (gold False) ---
_AGREEMENT: list[tuple[str, str, str]] = [
    ("Albert Einstein developed general relativity.", "General relativity was formulated by Einstein.", "Who developed general relativity?"),
    ("Gold's chemical symbol is Au.", "On the periodic table, gold is denoted Au.", "What is the chemical symbol for gold?"),
    ("The Pacific is the largest ocean on Earth.", "No ocean is larger than the Pacific.", "What is the largest ocean?"),
    ("Hamlet was written by William Shakespeare.", "Shakespeare is the author of Hamlet.", "Who wrote Hamlet?"),
    ("The Sahara is the largest hot desert.", "No hot desert is larger than the Sahara.", "What is the largest hot desert?"),
    ("Water is composed of hydrogen and oxygen.", "Water molecules consist of oxygen and hydrogen atoms.", "What is water composed of?"),
    ("Tokyo is the capital of Japan.", "Japan's capital city is Tokyo.", "What is the capital of Japan?"),
    ("The heart pumps blood through the body.", "Blood is circulated through the body by the heart.", "What organ pumps blood through the body?"),
]

# --- unit-equivalent agreement (gold False): (subject, val_a, val_b, question) ---
_UNIT_AGREEMENT: list[tuple[str, str, str, str]] = [
    ("A marathon", "42.195 kilometres", "about 26.2 miles", "How long is a marathon?"),
    ("Water freezes", "at 0 degrees Celsius", "at 32 degrees Fahrenheit", "At what temperature does water freeze?"),
    ("The Great Wall of China", "over 21,000 kilometres", "more than 13,000 miles", "How long is the Great Wall of China?"),
    ("The cruising altitude", "about 11,000 metres", "around 36,000 feet", "What is a typical cruising altitude?"),
    ("An adult's normal body temperature", "about 37 degrees Celsius", "about 98.6 degrees Fahrenheit", "What is normal body temperature?"),
]

# --- negation agreement (gold False): double negation / equivalent restatement ---
_NEGATION_AGREEMENT: list[tuple[str, str, str]] = [
    ("It is not true that the Earth is flat.", "The Earth is round.", "What is the shape of the Earth?"),
    ("It is false that the Sun orbits the Earth.", "The Earth orbits the Sun.", "What orbits what in the Solar System?"),
    ("It is not the case that whales are fish.", "Whales are mammals.", "Are whales fish or mammals?"),
    ("It is incorrect that the heart is in the abdomen.", "The heart is located in the chest.", "Where is the heart located?"),
]

# --- topic shift (gold False): one passage does not answer the question ---
_TOPIC_SHIFT: list[tuple[str, str, str]] = [
    ("The capital of Australia is Canberra.", "Sydney is the most populous city in Australia.", "What is the capital of Australia?"),
    ("Photosynthesis occurs in the chloroplasts of plant cells.", "The French Revolution began in 1789.", "Where does photosynthesis occur?"),
    ("Mitochondria produce ATP in cells.", "Ribosomes synthesize proteins in cells.", "What produces ATP in cells?"),
    ("Canberra is the capital of Australia.", "Australia is both a country and a continent.", "What is the capital of Australia?"),
    ("Water boils at 100 degrees Celsius at sea level.", "Ice is the solid form of water.", "At what temperature does water boil at sea level?"),
    ("The Nile flows through northeastern Africa.", "The Amazon rainforest is in South America.", "Through which region does the Nile flow?"),
    ("Shakespeare was an English playwright.", "Mozart was an Austrian composer.", "What was Shakespeare's profession?"),
    ("The speed of sound in air is about 343 metres per second.", "Sound cannot travel through a vacuum.", "What is the speed of sound in air?"),
]


def build_probe_set() -> list[ContextProbe]:
    """Assemble the full expanded probe set (deterministic)."""
    probes: list[ContextProbe] = []
    for a, b, q in _SENSE_TRAPS:
        probes.append((a, b, q, False, "sense_trap"))
    for a, b, q in _EXPLICIT:
        probes.append((a, b, q, True, "explicit"))
    for subj, va, vb, unit, q in _NUMERIC:
        probes.append((f"{subj} is {va} {unit}.", f"{subj} is {vb} {unit}.", q, True, "numeric"))
    for event, ya, yb, q in _TEMPORAL:
        probes.append((f"{event} in {ya}.", f"{event} in {yb}.", q, True, "temporal"))
    for a, b, q in _IMPLICIT:
        probes.append((a, b, q, True, "implicit"))
    for a, b, q in _AGREEMENT:
        probes.append((a, b, q, False, "agreement"))
    for subj, va, vb, q in _UNIT_AGREEMENT:
        probes.append((f"{subj} is {va}.", f"{subj} is {vb}.", q, False, "unit_agreement"))
    for a, b, q in _NEGATION_AGREEMENT:
        probes.append((a, b, q, False, "negation_agreement"))
    for a, b, q in _TOPIC_SHIFT:
        probes.append((a, b, q, False, "topic_shift"))
    return probes


def category_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for *_rest, cat in build_probe_set():
        counts[cat] = counts.get(cat, 0) + 1
    return counts
