"""Generate teaching sentences for three extension-eval categories.

Writes corpus/extension/{opposites,categories,negation}.txt. Each line is one
training passage. Multi-sentence examples are written WITHOUT a space after the
period (e.g. "the cup is not green .it is white .") because the notebook loader
splits passages at ". " boundaries; the word tokenizer still produces the same
tokens either way, so the model sees "green . it is white" as one passage.

Separation rules (checked independently by check_leakage.py; see README):
1. No line contains any complete sentence of a test prompt or its answer
   (e.g. "a robin is a bird .", "it is blue .", "she bought milk .").
2. Opposites: the three tested pairs never appear in a "the opposite of" wording,
   in either direction. They are taught only as "hot and cold are opposites ." etc.
3. Negation: the tested subjects box, door and ava never appear in a negation
   sentence; no line containing "not" pairs a tested pair (red/blue, open/closed,
   tea/milk); and no line contains two or more details of one test story (box, red,
   blue / door, open-or-closed / ava, buy, tea, milk). open/closed are a real
   antonym pair, so the opposites file may state that fact, but only in the plain
   "are opposites" / "the opposite of" wordings, never with "not" or "it is".
4. Categories: real-world facts (a salmon is a fish) are taught only in other
   wordings, as the assignment's surgeon -> patient example allows.
Vocabulary needed to make eval cases scorable (e.g. "robin", "box", "ava", or the
distractor "fabric") is taught in different sentences.
"""
import random
import re
from pathlib import Path

rng = random.Random(7)
OUT = Path(__file__).resolve().parent / "corpus" / "extension"

# Rule 1: every complete sentence of the extended tests' prompts, plus each answer sentence.
TEST_SENTENCES = {
    "the box is not red .", "it is blue .", "the box is blue .",
    "ava did not buy tea .", "she bought milk .", "ava bought milk .",
    "the door is not open .", "it is closed .", "the door is closed .",
    "a robin is a bird .", "a salmon is a fish .",
    "a puppy grows into a dog .", "a kitten grows into a cat .",
    "a carrot is a vegetable .", "an apple is a fruit .",
    "the opposite of hot is cold .", "the opposite of empty is full .",
    "the opposite of noisy is quiet .",
}
# Rule 3: each negation test story's details, one set per slot (open/closed is one
# slot because the pair is general knowledge, not an arbitrary story value).
NEGATION_TESTS = [
    [{"box"}, {"red"}, {"blue"}],
    [{"door"}, {"open", "closed"}],
    [{"ava"}, {"buy", "bought"}, {"tea"}, {"milk"}],
]
NEGATION_TEST_PAIRS = [{"red", "blue"}, {"open", "closed"}, {"tea", "milk"}]
NEGATION_TEST_SUBJECTS = {"box", "door", "ava"}


def words(line):
    return set(re.findall(r"\w+", line))


def sentences(line):
    return [s.strip() + " ." for s in line.split(".") if s.strip()]


def shares_test_values(line):
    present = words(line)
    return any(sum(1 for slot in test if present & slot) >= 2 for test in NEGATION_TESTS)


def allowed(line, negation_sentence=False):
    if any(s in TEST_SENTENCES for s in sentences(line)):
        return False
    if shares_test_values(line):
        return False
    if "not" in words(line) and any(pair <= words(line) for pair in NEGATION_TEST_PAIRS):
        return False
    if negation_sentence and words(line) & NEGATION_TEST_SUBJECTS:
        return False
    return True


def article(word):
    return "an" if word[0] in "aeiou" else "a"


# ---------------------------------------------------------------- opposites
PAIRS = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet"), ("big", "small"),
         ("tall", "short"), ("fast", "slow"), ("warm", "cool"), ("heavy", "light"),
         ("early", "late"), ("soft", "hard"), ("loud", "silent"), ("round", "flat"),
         ("wet", "dry"), ("up", "down"), ("clean", "dirty"), ("bright", "dark"),
         ("happy", "sad"), ("wide", "narrow"), ("strong", "weak"), ("long", "brief"),
         ("thick", "thin"), ("rich", "poor"), ("young", "old"), ("high", "low"),
         ("sweet", "sour"), ("near", "far"), ("deep", "shallow"), ("smooth", "rough"),
         ("open", "closed"), ("sharp", "dull")]
# Rule 2: tested pairs are never written with "the opposite of", in either direction.
TESTED_OPPOSITE_PAIRS = {frozenset(p) for p in [("hot", "cold"), ("empty", "full"), ("noisy", "quiet")]}
# box and door are tested negation subjects; they appear here (never in negation
# sentences) so the words exist in the vocabulary.
THINGS = ["room", "cup", "street", "sky", "bag", "river", "voice", "path", "coat", "hill", "box", "door"]


def opposites():
    lines = []
    for x, y in PAIRS:
        tested = frozenset((x, y)) in TESTED_OPPOSITE_PAIRS
        for a, b in [(x, y), (y, x)]:
            if not tested:
                lines.append(f"the opposite of {a} is {b} .")
                lines.append(f"{a} is the opposite of {b} .")
            lines.append(f"{a} and {b} are opposites .")
            if {a, b} == {"open", "closed"}:  # plain fact wordings only (rule 3)
                continue
            lines.append(f"if it is not {a} then it is {b} .")
            lines.append(f"{a} means not {b} .")
            lines.append(f"we say {a} when we do not mean {b} .")
            for thing in rng.sample(THINGS, 4):
                lines.append(f"the {thing} was {a} but now it is {b} .")
                lines.append(f"one {thing} is {a} and the other {thing} is {b} .")
    return [line for line in lines if allowed(line)]


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
YOUNG = [("puppy", "dog"), ("kitten", "cat"), ("lamb", "sheep"), ("calf", "cow"),
         ("foal", "horse"), ("chick", "hen"), ("duckling", "duck"), ("kid", "goat"),
         ("cub", "bear"), ("piglet", "pig")]


def categories():
    lines = []
    members = [(m, c) for c, ms in CATEGORIES.items() for m in ms]
    for m, c in members:
        lines.append(f"{article(m)} {m} is {article(c)} {c} .")
        lines.append(f"the {m} is a kind of {c} .")
        lines.append(f"{m} belongs to the {c} group .")
        lines.append(f"every {m} is one type of {c} .")
    # two-sentence analogies: "a X is a C1.a Y is a C2." with random other members
    for _ in range(420):
        (m1, c1), (m2, c2) = rng.sample(members, 2)
        lines.append(f"{article(m1)} {m1} is {article(c1)} {c1} .{article(m2)} {m2} is {article(c2)} {c2} .")
    # same-category pairs
    for c, ms in CATEGORIES.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                lines.append(f"the {ms[i]} and the {ms[j]} are both {c}s ." if c != "fish" else
                             f"the {ms[i]} and the {ms[j]} are both fish .")
    for y, a in YOUNG:
        lines.append(f"{article(y)} {y} grows into {article(a)} {a} .")
        lines.append(f"{article(y)} {y} is a young {a} .")
        lines.append(f"the {y} will become {article(a)} {a} one day .")
        lines.append(f"{article(a)} {a} starts life as {article(y)} {y} .")
        lines.append(f"a young {a} is called {article(y)} {y} .")
    for _ in range(120):
        (y1, a1), (y2, a2) = rng.sample(YOUNG, 2)
        lines.append(f"{article(y1)} {y1} grows into {article(a1)} {a1} .{article(y2)} {y2} grows into {article(a2)} {a2} .")
    return [line for line in lines if allowed(line)]


# ----------------------------------------------------------------- negation
OBJECTS = ["cup", "ball", "hat", "shirt", "wall", "chair", "lamp", "coat", "window", "gate"]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "brown", "pink", "grey"]
STATES = [("open", "closed"), ("locked", "unlocked"), ("wet", "dry"), ("full", "empty"),
          ("clean", "dirty"), ("wide", "narrow"), ("broken", "fixed"), ("here", "missing"),
          ("on", "off"), ("hot", "cold")]
PEOPLE = [("ava", "she"), ("ben", "he"), ("chloe", "she"), ("dan", "he"), ("eli", "he"),
          ("grace", "she"), ("hugo", "he"), ("ivy", "she"), ("jack", "he"), ("kate", "she")]
ITEMS = ["tea", "milk", "rice", "bread", "soup", "juice", "coffee", "cheese", "eggs",
         "butter", "water", "fish", "cake", "honey"]
VERBS = [("buy", "bought"), ("want", "wanted"), ("eat", "ate"), ("choose", "chose"),
         ("order", "ordered"), ("drink", "drank"), ("take", "took"), ("pick", "picked")]


def negation():
    negated, plain = [], []
    for obj in OBJECTS:
        for c1 in COLORS:
            for c2 in rng.sample([c for c in COLORS if c != c1], 2):
                negated.append(f"the {obj} is not {c1} .it is {c2} .the {obj} is {c2} .")
                negated.append(f"the {obj} is not {c1} , the {obj} is {c2} .")
        for s1, s2 in STATES:
            for a, b in [(s1, s2), (s2, s1)]:
                negated.append(f"the {obj} is not {a} .it is {b} .the {obj} is {b} .")
                negated.append(f"the {obj} is not {a} , the {obj} is {b} .")
    for name, pro in PEOPLE:
        for v, vp in VERBS:
            for i1 in rng.sample(ITEMS, 3):
                i2 = rng.choice([i for i in ITEMS if i != i1])
                negated.append(f"{name} did not {v} {i1} .{pro} {vp} {i2} .{name} {vp} {i2} .")
                negated.append(f"{name} did not {v} {i1} , {pro} {vp} {i2} instead .")
                negated.append(f"{i1} was not what {name} {vp} .{name} {vp} {i2} .")
            # plain statements (no negation) give every name, including ava, a base case
            for item in rng.sample(ITEMS, 2):
                plain.append(f"{name} {vp} {item} .")
    return ([line for line in negated if allowed(line, negation_sentence=True)]
            + [line for line in plain if allowed(line)])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("opposites", opposites), ("categories", categories), ("negation", negation)]:
        lines = sorted(set(fn()))
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(lines)} unique lines")
