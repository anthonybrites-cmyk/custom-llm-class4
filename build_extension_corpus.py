"""Generate teaching sentences for three extension-eval categories.

Writes corpus/extension/{opposites,categories,negation}.txt. Each line is one
training passage. Multi-sentence examples are written WITHOUT a space after the
period (e.g. "the cup is not green.it is white.") because the notebook loader
splits passages at ". " boundaries; the word tokenizer still produces the same
tokens either way, so the model sees "green . it is white" as one passage.

Deliberately excluded (see README): the 48 eval prompts, their answer sentences,
and the tested pairs in the tested direction. The model must learn the pattern
from OTHER examples. Vocabulary needed to make eval cases scorable (e.g. the word
"robin" or the distractor "fabric") is taught in different sentences.
"""
import random
from pathlib import Path

rng = random.Random(7)
OUT = Path(__file__).resolve().parent / "corpus" / "extension"


def article(word):
    return "an" if word[0] in "aeiou" else "a"


# ---------------------------------------------------------------- opposites
# (x, y) pairs. Tested pairs hot/cold, empty/full, noisy/quiet are taught, but
# NEVER as "the opposite of <tested word> is <answer>" (that is the test item).
PAIRS = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet"), ("big", "small"),
         ("tall", "short"), ("fast", "slow"), ("warm", "cool"), ("heavy", "light"),
         ("early", "late"), ("soft", "hard"), ("loud", "silent"), ("round", "flat"),
         ("wet", "dry"), ("up", "down"), ("clean", "dirty"), ("bright", "dark"),
         ("happy", "sad"), ("wide", "narrow"), ("strong", "weak"), ("long", "brief"),
         ("thick", "thin"), ("rich", "poor"), ("young", "old"), ("high", "low"),
         ("sweet", "sour"), ("near", "far"), ("deep", "shallow"), ("smooth", "rough"),
         ("open", "closed"), ("sharp", "dull")]
TESTED_OPPOSITE_PROMPT_WORDS = {"hot", "empty", "noisy"}
THINGS = ["room", "cup", "street", "sky", "bag", "river", "voice", "path", "coat", "hill"]


def opposites():
    lines = []
    for x, y in PAIRS:
        for a, b in [(x, y), (y, x)]:
            if a not in TESTED_OPPOSITE_PROMPT_WORDS:  # skip the exact test direction
                lines.append(f"the opposite of {a} is {b} .")
            lines.append(f"{a} and {b} are opposites .")
            lines.append(f"{a} is the opposite of {b} .")
            lines.append(f"if it is not {a} then it is {b} .")
            lines.append(f"{a} means not {b} .")
            lines.append(f"we say {a} when we do not mean {b} .")
            for thing in rng.sample(THINGS, 4):
                lines.append(f"the {thing} was {a} but now it is {b} .")
                lines.append(f"one {thing} is {a} and the other {thing} is {b} .")
    return lines


# --------------------------------------------------------------- categories
CATEGORIES = {
    "bird": ["sparrow", "robin", "eagle", "owl", "crow", "parrot", "duck", "hen"],
    "fish": ["trout", "salmon", "tuna", "shark", "cod", "herring"],
    "vegetable": ["carrot", "broccoli", "onion", "potato", "spinach", "cabbage"],
    "fruit": ["apple", "banana", "orange", "pear", "peach", "mango", "grape", "cherry"],
    "tool": ["hammer", "saw", "wrench", "drill", "chisel"],
    "vehicle": ["car", "bus", "train", "truck", "taxi", "bicycle", "van"],
    "metal": ["iron", "copper", "gold", "silver", "steel"],
    "fabric": ["cotton", "silk", "wool", "linen", "denim"],
    "tree": ["oak", "pine", "maple", "birch", "cedar"],
    "animal": ["dog", "cat", "horse", "goat", "cow", "sheep", "pig", "bear"],
}
# The three tested facts (salmon->fish, kitten->cat, apple->fruit) are never
# written as the plain "a X is a Y ." answer sentence; they appear in other frames.
TESTED_FACTS = {("salmon", "fish"), ("apple", "fruit")}
YOUNG = [("puppy", "dog"), ("kitten", "cat"), ("lamb", "sheep"), ("calf", "cow"),
         ("foal", "horse"), ("chick", "hen"), ("duckling", "duck"), ("kid", "goat"),
         ("cub", "bear"), ("piglet", "pig")]
TESTED_YOUNG = {("kitten", "cat")}


def categories():
    lines = []
    members = [(m, c) for c, ms in CATEGORIES.items() for m in ms]
    for m, c in members:
        if (m, c) not in TESTED_FACTS:
            lines.append(f"{article(m)} {m} is {article(c)} {c} .")
        lines.append(f"the {m} is a kind of {c} .")
        lines.append(f"{m} belongs to the {c} group .")
        lines.append(f"every {m} is one type of {c} .")
    # two-sentence analogies: "a X is a C1.a Y is a C2." with random other members
    for _ in range(420):
        (m1, c1), (m2, c2) = rng.sample(members, 2)
        if (m2, c2) in TESTED_FACTS or (m1, c1) in TESTED_FACTS:
            continue
        lines.append(f"{article(m1)} {m1} is {article(c1)} {c1} .{article(m2)} {m2} is {article(c2)} {c2} .")
    # same-category pairs
    for c, ms in CATEGORIES.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                lines.append(f"the {ms[i]} and the {ms[j]} are both {c}s ." if c != "fish" else
                             f"the {ms[i]} and the {ms[j]} are both fish .")
    for y, a in YOUNG:
        if (y, a) not in TESTED_YOUNG:
            lines.append(f"{article(y)} {y} grows into {article(a)} {a} .")
        lines.append(f"{article(y)} {y} is a young {a} .")
        lines.append(f"the {y} will become {article(a)} {a} one day .")
        lines.append(f"{article(a)} {a} starts life as {article(y)} {y} .")
    for _ in range(120):
        (y1, a1), (y2, a2) = rng.sample(YOUNG, 2)
        if (y1, a1) in TESTED_YOUNG or (y2, a2) in TESTED_YOUNG:
            continue
        lines.append(f"{article(y1)} {y1} grows into {article(a1)} {a1} .{article(y2)} {y2} grows into {article(a2)} {a2} .")
    return lines


# ----------------------------------------------------------------- negation
OBJECTS = ["cup", "ball", "hat", "shirt", "wall", "chair", "lamp", "coat", "box", "door", "window", "gate"]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "brown", "pink", "grey"]
STATES = [("open", "closed"), ("locked", "unlocked"), ("wet", "dry"), ("full", "empty"),
          ("clean", "dirty"), ("wide", "narrow"), ("broken", "fixed"), ("here", "missing"),
          ("on", "off"), ("hot", "cold")]
# Never reuse a tested object with its tested attribute (box/red/blue, door/open/closed).
BANNED = {("box", "red"), ("box", "blue"), ("door", "open"), ("door", "closed")}
PEOPLE = [("ava", "she"), ("ben", "he"), ("chloe", "she"), ("dan", "he"), ("eli", "he"),
          ("grace", "she"), ("hugo", "he"), ("ivy", "she"), ("jack", "he"), ("kate", "she")]
ITEMS = ["tea", "milk", "rice", "bread", "soup", "juice", "coffee", "cheese", "eggs",
         "butter", "water", "fish", "cake", "honey"]
VERBS = [("buy", "bought"), ("want", "wanted"), ("eat", "ate"), ("choose", "chose"),
         ("order", "ordered"), ("drink", "drank"), ("take", "took"), ("pick", "picked")]
# ava never appears with tea, milk or buy (the test story).
BANNED_PERSON = {("ava", "tea"), ("ava", "milk"), ("ava", "buy")}


def negation():
    lines = []
    for obj in OBJECTS:
        for c1 in COLORS:
            for c2 in rng.sample([c for c in COLORS if c != c1], 2):
                if (obj, c1) in BANNED or (obj, c2) in BANNED:
                    continue
                lines.append(f"the {obj} is not {c1} .it is {c2} .the {obj} is {c2} .")
                lines.append(f"the {obj} is not {c1} , the {obj} is {c2} .")
        for s1, s2 in STATES:
            for a, b in [(s1, s2), (s2, s1)]:
                if (obj, a) in BANNED or (obj, b) in BANNED:
                    continue
                lines.append(f"the {obj} is not {a} .it is {b} .the {obj} is {b} .")
                lines.append(f"the {obj} is not {a} , the {obj} is {b} .")
    for name, pro in PEOPLE:
        for v, vp in VERBS:
            for i1 in rng.sample(ITEMS, 3):
                i2 = rng.choice([i for i in ITEMS if i != i1])
                if any((name, w) in BANNED_PERSON for w in (i1, i2, v)):
                    continue
                lines.append(f"{name} did not {v} {i1} .{pro} {vp} {i2} .{name} {vp} {i2} .")
                lines.append(f"{name} did not {v} {i1} , {pro} {vp} {i2} instead .")
                lines.append(f"{i1} was not what {name} {vp} .{name} {vp} {i2} .")
    return lines


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("opposites", opposites), ("categories", categories), ("negation", negation)]:
        lines = sorted(set(fn()))
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(lines)} unique lines")
