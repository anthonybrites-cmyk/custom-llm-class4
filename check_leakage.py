"""Check a training corpus against the 48 fixed evals, beyond the notebook's exact-prefix check.

Usage:
  python check_leakage.py                                       # scan corpus/ (my teaching files)
  python check_leakage.py --corpus experiments/extension/run/corpus.txt   # full training text of a run

Fails (exit 1) if any line contains:
  A. an eval prompt (normalized tokens),
  B. any complete sentence of an eval prompt or its answer sentence (prompt's last clause + answer),
  C. two or more details of one negation test story, or "not" together with a tested value pair,
  D. a tested opposite pair in an "opposite of" wording, in either direction.
It also reports, for each extend_corpus case, the longest run of tokens shared word for word with
any training line, so a reader can see how close the nearest teaching sentence is. Starter cases
are handled by the notebook, which removes classroom sentences containing their prompts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent / "evals" / "language_evals.json"
NEGATION_VALUES = {  # per case: story details, one set per slot (inflections grouped;
    # open/closed is one slot because the antonym pair is general knowledge)
    "lang_31": [{"box"}, {"red"}, {"blue"}],
    "lang_32": [{"ava"}, {"buy", "bought"}, {"tea"}, {"milk"}],
    "lang_33": [{"door"}, {"open", "closed"}],
}
NEGATION_PAIRS = {"lang_31": {"red", "blue"}, "lang_32": {"tea", "milk"}, "lang_33": {"open", "closed"}}
TESTED_OPPOSITES = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet")]


def tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower())


def sentences(toks):
    out, current = [], []
    for t in toks:
        current.append(t)
        if t == ".":
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current + ["."]))
    return out


def longest_shared_run(a, b):
    best, end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        row = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                row[j] = prev[j - 1] + 1
                if row[j] > best:
                    best, end = row[j], i
        prev = row
    return best, a[end - best:end]


def load_lines(corpus):
    paths = [corpus] if corpus.is_file() else sorted(
        p for p in corpus.rglob("*") if p.is_file() and p.suffix in {".txt", ".md"}
        and p != corpus / "README.md" and not any(part.startswith(".") for part in p.parts))
    lines = []
    for path in paths:
        lines += [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return paths, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    cases = json.loads(SUITE.read_text(encoding="utf-8"))["cases"]
    paths, raw = load_lines(args.corpus)
    lines = [tokens(l) for l in raw]
    joined = [" ".join(t) for t in lines]
    line_sentences = [set(sentences(t)) for t in lines]
    words = [set(t) for t in lines]
    print(f"Scanned {len(raw):,} lines from {', '.join(str(p) for p in paths)}")

    failures = 0
    print("\nA/B/C/D: lines that match (all must be 0)")
    for case in cases:
        prompt = tokens(case["prompt"])
        full = prompt + [case["answer"], "."]
        a = sum(" ".join(prompt) in j for j in joined)
        test_sentences = set(sentences(full))
        b = sum(bool(s & test_sentences) for s in line_sentences)
        c = sum(sum(1 for slot in NEGATION_VALUES[case["id"]] if w & slot) >= 2
                or ("not" in w and NEGATION_PAIRS[case["id"]] <= w)
                for w in words) if case["id"] in NEGATION_VALUES else 0
        d = 0
        if case["category"] == "opposites":
            x, y = next(p for p in TESTED_OPPOSITES if case["answer"] in p)
            d = sum("opposite of" in j and {x, y} <= w for j, w in zip(joined, words))
        failures += a + b + c + d
        if case["group"] == "extend_corpus" or a or b:
            print(f"  {case['id']:8} A={a} B={b} C={c} D={d}  {case['prompt']} -> {case['answer']}")

    print("\nClosest teaching line for each extend_corpus case (longest word-for-word run shared with prompt + answer)")
    for case in cases:
        if case["group"] != "extend_corpus":
            continue
        full = tokens(case["prompt"]) + [case["answer"]]
        best = max(((longest_shared_run(full, t), l) for t, l in zip(lines, raw)), key=lambda x: x[0][0])
        (length, run), line = best
        print(f"  {case['id']:8} {length:2}/{len(full)} tokens  shared: \"{' '.join(run)}\"  e.g. {line!r}")

    print("\nRESULT:", "PASS - no eval prompt, test sentence, negation-story values or tested opposite frame"
          if failures == 0 else f"FAIL - {failures} matching lines")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
