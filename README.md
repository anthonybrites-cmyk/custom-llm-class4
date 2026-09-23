# Class 4 — Building a Custom LLM (nanoGPT, word tokens)

Anthony Brites · From Zero to AI Agents, Fall 26

I trained Karpathy's nanoGPT (2 blocks, 4 heads, 64-dim embeddings, 48-token context,
~112k–128k parameters) from scratch on CPU, twice:

1. **Starter experiment** — the supplied classroom corpus only.
2. **Corpus-extension experiment** — the classroom corpus plus 3,116 teaching passages I
   generated for three of the eval categories: **opposites**, **categories_and_analogies**,
   and **negation**. No eval prompt, test sentence or test-story detail is in them; a separate
   script checks this (see [Keeping the tests out of training](#keeping-the-tests-out-of-training)).

Both runs use the same 48 fixed language evals (unchanged), run before and after training.
This is a tiny model that continues sentences from a narrow corpus. It is not a chat
assistant and the numbers below are a development benchmark, not a claim of general ability.

The starter notebook, nanoGPT source, eval runner and chat script come from the
[course sample project](https://github.com/pepealonso95/custom-llm); its original README is
kept as [SAMPLE_PROJECT_README.md](SAMPLE_PROJECT_README.md).

## Repository map

| What | Where |
|---|---|
| Executed notebook — starter experiment | [experiments/starter/custom_llm.executed.ipynb](experiments/starter/custom_llm.executed.ipynb) |
| Executed notebook — extension experiment (also the current `custom_llm.ipynb`) | [experiments/extension/custom_llm.executed.ipynb](experiments/extension/custom_llm.executed.ipynb) |
| Full results folders (config, corpus, split, tokenization, inspection, history, samples, model.pt, evals, ZIP) | [experiments/starter/run/](experiments/starter/run/) · [experiments/extension/run/](experiments/extension/run/) |
| Fixed 48-case eval suite (unchanged, SHA-256 `e8affcd7…` as pinned by the notebook) | [evals/language_evals.json](evals/language_evals.json) · [evals/README.md](evals/README.md) |
| Eval runner / chat interface | [run_evals.py](run_evals.py) · [chat.py](chat.py) |
| Generator for my teaching sentences | [build_extension_corpus.py](build_extension_corpus.py) → [corpus/extension/](corpus/extension/) |
| Leakage check (my script) and its reports on each run's full training text | [check_leakage.py](check_leakage.py) → [starter](experiments/starter/leakage_check.txt) · [extension](experiments/extension/leakage_check.txt) |
| Chat evidence | [experiments/extension/chat_transcript.json](experiments/extension/chat_transcript.json) · [chat_screenshot.png](experiments/extension/chat_screenshot.png) · [chat_session.exp](chat_session.exp) |
| Not my results | `examples/` and `legacy/` are the course sample's own reference outputs, kept unchanged from the upstream repo |

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace custom_llm.ipynb   # or open it in Jupyter and Run All
```

Every Run All creates a new `llm_runs/<timestamp>/` folder and ZIP. I ran the notebook
headlessly with `nbconvert` on macOS; the notebook ran with no errors in both experiments.
One local gotcha: `run_evals.model_hash()` calls `tensor.numpy()`, and the original
`requirements.txt` did not list numpy (Colab preinstalls it). I added it.

To reproduce the extension corpus: `python build_extension_corpus.py` regenerates the three
files in `corpus/extension/` deterministically (fixed random seed). Running the notebook as
committed reproduces the **extension** experiment. To reproduce the **starter** experiment,
move `corpus/extension/` out of `corpus/` first (so `corpus/` holds only its README), run
the notebook, then move it back.

To check a corpus against the evals (both exit with an error if anything matches):

```bash
.venv/bin/python check_leakage.py                                              # my teaching files in corpus/
.venv/bin/python check_leakage.py --corpus experiments/extension/run/corpus.txt   # everything a run trained on
```

Before the starter run I ran a 10-step setup check with the same settings otherwise. It
completed all 10 steps with no interruption (panel loss 4.93 → 4.21). I did not use it as
evidence, and it is not committed. Neither of the two runs below was interrupted or failed.
An earlier version of the extension run was **superseded**: its teaching lines were too close to
some tests, so I removed them and reran (details under
[Keeping the tests out of training](#keeping-the-tests-out-of-training)). Only the rerun is used
as evidence.

## My three choices and my prediction

**Corpus.** Experiment 1: `CORPUS = "classroom"` with an empty `corpus/`. Experiment 2: the
same setting with my three generated `.txt` files in `corpus/extension/`. All source text is
synthetic and written by me/my generator — no external documents, no third-party text, so
there are no permission limits on sharing it; and no PDFs, so there were no extraction
warnings to resolve (both manifests list 0 warnings and 0 ignored files). The manifest for each run is in `corpus_manifest.json`
([starter](experiments/starter/run/corpus_manifest.json) ·
[extension](experiments/extension/run/corpus_manifest.json)).

**Training steps = 3,000.** One step is one AdamW update on 32 sampled passages, not an
epoch. 3,000 × 32 ≈ 96,000 passage views ≈ 23 passes over the 4,132 starter training
passages. I kept the class baseline for comparability; training time was never a constraint
(about 10 seconds per run on an Apple M4).

**Learning rate = 0.001.** The conventional AdamW starting point for a model this size. The
notebook warms up linearly over 100 steps and cosine-decays to 10% of this value. Too large
(say 0.1) and updates overshoot: loss can spike or go non-finite and embeddings jump instead
of settling. Too small (say 0.00001) and each step barely moves the weights, so 3,000 steps
end with loss still far above what the model could reach.

**What I predicted before running** (verbatim text is in the notebook's "My prediction" cell):

| Prediction (exp. 1) | What actually happened |
|---|---|
| Step-0 loss ≈ ln(136) ≈ 4.9 on both panels | 4.926 train / 4.928 validation ✔ |
| Final training loss 1.0–1.5 | **0.678** — lower than I guessed; the eight templates are more predictable than I estimated |
| Validation within ~0.2 of training | 0.706 vs 0.678 (gap 0.03) ✔ |
| Halfway samples already template-shaped | Yes — and almost nothing changed between step 1,500 and 3,000 |
| `customer`'s neighbors become the other customer nouns | shopper .978, client .977, buyer .977, subscriber .971, consumer .970 ✔ |
| Evals: ~9/48 untrained → most starter_patterns, ~half transfer, 0/24 extension | 9/48 → 20/48: 16/16, 4/8, 0/24 ✔ |
| (not predicted) | Temperature 0.3 / 0.8 / 1.2 produced nearly identical samples — the model is so peaked that temperature barely matters under a fixed seed |

The notebook's prediction cell also holds my experiment 2 prediction, plus a note written before
the final (stricter) rerun. The rerun expectations against the result:

| Prediction (exp. 2, final rerun) | What actually happened |
|---|---|
| Coverage 24/48 → 33/48 scorable | 33/48 ✔ |
| Opposites 1–3 / 3 (tested pairs never appear with "the opposite of") | 3/3 ✔ — better than I expected |
| Categories 1–2 / 3 | 2/3 ✔ (kitten→cat failed again, now choosing `horse`) |
| Negation 0–1 / 3 | 0/3 ✔ — including door→closed, which the superseded version had passed |
| Starter cases might get slightly worse | They did not: 16/16, and transfer 4/8 → 7/8 (noise, see below) |
| Total probably below 27/48 | 28/48 ✘ — negation fell as expected, but the noisy transfer flips went the other way |

## The runs

| | Starter | Extension |
|---|---|---|
| Completed steps / interrupted | 3,000 / no | 3,000 / no |
| Elapsed | 8.55 s | 10.54 s |
| Hardware | Apple M4, macOS 26.3.1, CPU, PyTorch 2.14.0, Python 3.11.2 | same |
| Parameters | 111,872 | 127,936 (larger embedding table) |
| Generated classroom sentences → reserved → unique | 6,360 → 160 withheld (contain the 16 reserved test prefixes) → 4,592 unique | same base + 3,116 new unique passages = 7,708 unique |
| Train / validation split (90/10 by unique passage) | 4,132 / 460 | 6,937 / 771 |
| Vocabulary (incl. UNK/BOS/EOS) | 136 | 387 |
| Training / held-out unknown-token rate | 0.00% / 0.00% | 0.00% / 0.00% |
| Types omitted by the 509-type cap | 0 | 0 |
| Loss panels | 20 training + 20 validation docs | 20 + 20 |
| Files | [config](experiments/starter/run/config.json) · [training_summary](experiments/starter/run/training_summary.json) · [vocabulary_report](experiments/starter/run/vocabulary_report.json) · [training.csv](experiments/starter/run/training.csv) | [config](experiments/extension/run/config.json) · [training_summary](experiments/extension/run/training_summary.json) · [vocabulary_report](experiments/extension/run/vocabulary_report.json) · [training.csv](experiments/extension/run/training.csv) |

Both splits keep 0% unknown tokens because every word type fits under the 509 cap. The split
is by passage, not source file, and validation passages come from the same templates as
training, so it tests whether the model learned the templates — not whether it can handle
sentence shapes it has never seen (the near-zero probabilities on the "transfer" evals show it
mostly cannot).

## Loss curves and the full loss table

These are **fixed panels of at most 20 training and 20 validation documents**, mean loss
over non-padding next-token targets. They are small estimates, not full-corpus numbers, and
the two experiments are **not comparable** with each other (different vocabulary sizes: an
untrained model starts near ln 136 ≈ 4.9 vs ln 387 ≈ 5.96).

Starter ([history.json](experiments/starter/run/history.json)):

![starter loss](experiments/starter/run/training_curves.svg)

| step | training loss | validation loss |
|---|---|---|
| 0 | 4.9263 | 4.9275 |
| 1,500 | 0.6821 | 0.7182 |
| 3,000 | 0.6783 | 0.7061 |

Extension ([history.json](experiments/extension/run/history.json)):

![extension loss](experiments/extension/run/training_curves.svg)

| step | training loss | validation loss |
|---|---|---|
| 0 | 6.0056 | 5.9791 |
| 1,500 | 0.8446 | 0.8715 |
| 3,000 | 0.7853 | 0.8158 |

In the extension run the validation panel starts slightly *below* the training panel (5.979 vs
6.006) and ends slightly above it (0.816 vs 0.785). Each panel is a different random draw of 20
documents, so gaps this small mostly show which documents were drawn, not generalization.

## Text samples: untrained, halfway, final

Same starting token, sampling seed 2026, temperature 0.8, four samples each. Full files:
[starter samples/](experiments/starter/run/samples/) · [extension samples/](experiments/extension/run/samples/).

**Starter, step 0** — random word salad, no sentence shape:
```
pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus question customer at mortgage nurse in instructor
kitchen purchase journey product question discussion journey service . nurse local
compared and purchase update mortgage question loan taste in market treatment learned another item bicycle product bicycle focused data and dentist recommended mango apple taxi bicycle delivery peach quality update student lesson
important hospital juice patient return recommended deposit tutor returned understand kitchen student design ordered hospital treatment important package traffic with yesterday investment important of mentioned store ordered mortgage nurse shopper the station
```
**Starter, step 1,500** — complete template sentences with matching domains:
```
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
we learned about the important website during a discussion of data .
our school has a question about the different instructor and course .
```
**Starter, step 3,000** — two of four samples are identical to step 1,500; the model had learned what it could by halfway:
```
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```
**Extension, step 0** — word salad again, now drawn from the larger 387-word vocabulary:
```
oak thin program narrow rich student flat when design local deep support loud hill update loud he is heavy hammer we therapist checking gate coffee wool drill closed gate learned grape service
is focused cup window green took small our parrot vehicle travel item tool system missing bank dry to near white fixed mentioned banana learning potato foal metals teacher foal juice onion nurse
purchase early school ball local cheese , small explains he website did platform hot professor slow apple purchase kind discussed software brand code mortgage application , spinach order say cake say want
silver mean school another bright cabbage locked mango brief , physician pink treatment birch application application light chisel cedar drank kitten copper small dog merchandise today birch to car kid ben station
```
**Extension, step 1,500** — all three of my new templates already have their shape, but the
content is not always right: `narrow … dry` is not an antonym pair, and the negation sample
contradicts itself:
```
the cup was narrow but now it is dry .
chloe did not drink juice , she drank water instead .
every mango is one type of fruit .
the shirt is not green . it is grey . the shirt is yellow .
```
**Extension, step 3,000** — with this seed, all four samples happen to be classroom templates:
```
the team discussed the orange and the harvest at the kitchen .
we learned about the local credit during a discussion of interest .
a review of fruit helped us understand the new peach .
a review of order helped us understand the important subscriber .
```
Visible change: from step 0 to 1,500 the output goes from unrelated words to whole template
sentences. Visible *lack* of change: between 1,500 and 3,000 the starter samples barely move
(two of four are identical), matching the flat loss curve. The extension's step-1,500 sample
`the shirt is not green . it is grey . the shirt is yellow .` is the most telling one: the
three-sentence negation shape is perfect, but the last word is not copied from the middle
sentence. The same failure shows up in the negation evals. None of the 24 saved samples is
empty; the only garbled ones are the step-0 samples.

## How the model learns — traced with my actual numbers

The five explanations below are in my own words (written during the session, lightly edited);
the numbers come from [tokenization.json](experiments/starter/run/tokenization.json) and
[inspection.json](experiments/starter/run/inspection.json) of the starter run (the extension
run's equivalents: [tokenization.json](experiments/extension/run/tokenization.json) ·
[inspection.json](experiments/extension/run/inspection.json)).

**One word, traced end to end.** The starter training sentence
`the customer ordered the product after checking the price .` is tokenized into 10 word/punctuation
tokens and encoded as `[1, 118, 28, 77, 118, 87, 6, 21, 118, 86, 3, 2]` — `<BOS>`=1, `the`=118,
`customer`=**28**, `ordered`=77, … `.`=3, `<EOS>`=2. ID 28 selects row 28 of the 136 × 64 embedding
table. That row is `customer`'s 64-number vector: before training it starts
`[-0.0576, -0.0048, 0.0426, 0.0193, 0.0156, -0.0288, …]`; after 3,000 updates it starts
`[0.0366, -0.0182, 0.1330, 0.1059, 0.0630, 0.0189, …]` (all 64 values, before and after, are in
inspection.json). The ID never changed; every one of the 64 numbers did.

**Corpus.** The corpus is the pile of example sentences the model reads. Eight templates × six
domains produce sentences such as `the team discussed the surgeon and the patient at the hospital .`
Nobody labels the domains; the model only ever sees next-word prediction targets.

**Tokens and IDs.** *"A token is a word or punctuation piece, and the ID is the number attached to
it. Umbrella gives the model trouble because it is unknown to the model and is lumped into the
unknown token bin."* Trace: the training document `today the school focused on lesson and the
local professor .` becomes IDs `[1, 121, 118, 101, 42, 74, 61, 7, 118, 63, 88, 3, 2]`. ID 1 is
`<BOS>`, 2 is `<EOS>`, `the` is 118 (it appears twice), `school` is 101. The numbers are row
labels, not quantities — 118 is not "more" than 101.

**Embeddings (vectors).** *"The embedding is 64 columns within a row that capture how a word is
used relative to how other words are used. Customer ended up next to shopper and client because
these words are all used similarly across the example sentences."* The embedding table is
136 × 64. `customer` is ID 28; row 28 starts `[-0.0576, -0.0048, 0.0426, …]` before training and
`[0.0366, -0.0182, 0.1330, …]` after (all 64 numbers are in inspection.json). Cosine neighbours
of that row before training: `bus .21, educator .20, helped .20` (random). After: `shopper .978,
client .977, buyer .977, subscriber .971, consumer .970`. Same for `surgeon` → `dentist, physician,
nurse, doctor, therapist`. The model was never told these are related; they occupy the same slots.

**Neural-network weights, prediction and probabilities.** *"The model gives probabilities for the
next word following the prompt, and those six verbs are very similar because in the training
corpus they are the ones that most often follow the prompt."* Prefix `the customer`:

| | top next-word probabilities |
|---|---|
| before training | customer .016, bus .011, educator .010, us .010, application .010 (≈ uniform 1/136) |
| after training | reviewed .178, recommended .171, ordered .169, selected .163, compared .160, returned .143 |

Those six verbs are exactly the six that follow "the customer" in the shopping template, each
about equally often, so they share ~99% of the probability.

**Loss, gradient and the saved weight update.** *"Loss is how bad a guess is compared to the
example sentences. The gradient measures how the loss would change if a weight moved slightly.
When the gradient is positive, loss would increase, so the weight is updated using the negative
of the gradient so that loss decreases and guessing gets better."* The notebook saved the very
first update for `customer`, coordinate 0:

| before | gradient | learning rate at step 1 | after |
|---|---|---|---|
| −0.0575919 | +0.000693 | 0.00001 (warmup: 1% of 0.001) | −0.0576019 |

The gradient is positive, so the value moved *down* (by 0.00001 — AdamW's first step is
roughly `lr × sign(gradient)`). After 3,000 such updates across all 111,872 parameters this
coordinate ends at +0.0366 and the loss falls from 4.93 to 0.68. The extension run shows the
opposite sign: `customer` is ID 88 there, its coordinate 0 started at 0.0273822 with gradient
**−0.000502**, and the update moved it *up* to 0.0273922 — same rule, opposite direction.

**Train-vs-validation gap.** *"The small gap shows that the model did learn a pattern and did not
simply memorize specific sentences. It does not tell you that the model learned alternate
sentence structures."*

**Attention.** The notebook prints head 1 of block 1 for the prefix `<BOS> the customer`. Trained
rows (each row is one position looking back at earlier positions):
`[1.0, 0, 0]`, `[0.606, 0.394, 0]`, `[0.485, 0.423, 0.092]`. The causal mask means position 3
(`customer`) can only attend to `<BOS>`, `the` and itself — never to later tokens — and here it
mostly looks back at the start of the sentence. Attention is how "the box is not red . it is
blue" can, in principle, influence the word after "the box is": the last position can read
earlier positions. Whether it *does* is what the negation evals test — and this 2-layer model
failed all three.

**Probabilities → generated words, and temperature.** Generation samples one word from the
probability distribution, appends it, and repeats until `<EOS>` or 32 tokens. Temperature divides
the logits before the softmax: below 1 sharpens the distribution toward the top word, above 1
flattens it. No weights change. The comparison uses the same `<BOS>` start token and the same
sampling seed (2026) at each temperature, on the final trained model.
[starter temperature_comparison.json](experiments/starter/run/temperature_comparison.json)
shows 0.3, 0.8 and 1.2 producing almost the same four sentences: the trained distribution is so
peaked (place names at ~0.98) that flattening by 1.2 rarely changes the sampled word. In the
[extension run](experiments/extension/run/temperature_comparison.json) 1.2 did change the output.
It produced one broken sentence, `we say deep when we do not mean did is new shopper hill is not`,
and one fluent sentence that contradicts itself, `coffee was not what dan wanted . dan wanted
coffee .`. The larger vocabulary leaves more probability on wrong words, and flattening lets the
sampler reach them. At 0.3 all four extension samples are classroom templates.

**What stayed fixed vs. what changed.** Fixed across both experiments: architecture, seed 42,
batch 32, block 48, 3,000 steps, LR schedule, eval panels of 20+20 (re-sampled from each run's
own split), generation seed 2026 and temperature 0.8, and the 48 eval cases. Changed during
training: only the weights. Changed only at inference: temperature and sampling seed.

## The 48 language evals

**Scoring rule** (from [run_evals.py](run_evals.py)): only the prompt is fed to the model — no
answer choices, no key. The runner reads the model's next-token probability for each of the four
single-word choices and scores 1 if the correct word has the highest probability, 0 otherwise;
ties score 0. If any prompt or choice word is not in the vocabulary, the case is
`out_of_vocabulary`: it counts as 0 in **all-case success** and is excluded from **scorable
accuracy**. **Coverage** is the fraction of cases that could be scored. Separately, the runner
saves a **free continuation** (temperature 0.8, seed 2026 + case index, ≤ 24 tokens), which is
*not* the score. The runner also asserts the model hash is unchanged after evaluation.

### Four-row comparison

| Experiment | Stage | Correct | All-case success | Scorable accuracy | Coverage | starter_patterns (16) | starter_transfer (8) | extend_corpus (24) | Files |
|---|---|---|---|---|---|---|---|---|---|
| Starter | untrained | 9/48 | 18.8% | 37.5% | 24/48 | 6/16 | 3/8 | 0/24 (0 scorable) | [summary](experiments/starter/run/language_evals/untrained/eval_summary.json) · [results.json](experiments/starter/run/language_evals/untrained/eval_results.json) · [csv](experiments/starter/run/language_evals/untrained/eval_results.csv) |
| Starter | trained | 20/48 | 41.7% | 83.3% | 24/48 | 16/16 | 4/8 | 0/24 (0 scorable) | [summary](experiments/starter/run/language_evals/final/eval_summary.json) · [results.json](experiments/starter/run/language_evals/final/eval_results.json) · [csv](experiments/starter/run/language_evals/final/eval_results.csv) |
| Extension | untrained | 7/48 | 14.6% | 21.2% | 33/48 | 3/16 | 3/8 | 1/24 (9 scorable) | [summary](experiments/extension/run/language_evals/untrained/eval_summary.json) · [results.json](experiments/extension/run/language_evals/untrained/eval_results.json) · [csv](experiments/extension/run/language_evals/untrained/eval_results.csv) |
| Extension | trained | **28/48** | **58.3%** | 84.8% | **33/48** | 16/16 | 7/8 | **5/24** (9 scorable) | [summary](experiments/extension/run/language_evals/final/eval_summary.json) · [results.json](experiments/extension/run/language_evals/final/eval_results.json) · [csv](experiments/extension/run/language_evals/final/eval_results.csv) |

Untrained scores are chance-level. Picking one of four at random is right ~25% of the time:
about 6 of the starter model's 24 scorable cases and 8 of the extension model's 33. The
untrained models got 9/24 and 7/33. Comparison files:
[starter](experiments/starter/run/language_eval_comparison.json) ·
[extension](experiments/extension/run/language_eval_comparison.json).

### By category

| Category (cases) | Starter untrained | Starter trained | Extended untrained | Extended trained |
|---|---|---|---|---|
| domain_context (8) | 3/8 (8 scorable) | 8/8 (8 scorable) | 1/8 (8 scorable) | 8/8 (8 scorable) |
| domain_place (8) | 3/8 (8 scorable) | 8/8 (8 scorable) | 2/8 (8 scorable) | 8/8 (8 scorable) |
| new_wording (8) | 3/8 (8 scorable) | 4/8 (8 scorable) | 3/8 (8 scorable) | 7/8 (8 scorable) |
| opposites (3) ★ | 0/3 (0 scorable) | 0/3 (0 scorable) | 1/3 (3 scorable) | **3/3** (3 scorable) |
| categories_and_analogies (3) ★ | 0/3 (0 scorable) | 0/3 (0 scorable) | 0/3 (3 scorable) | **2/3** (3 scorable) |
| negation (3) ★ | 0/3 (0 scorable) | 0/3 (0 scorable) | 0/3 (3 scorable) | **0/3** (3 scorable) |
| grammar (3) | 0/3 (0) | 0/3 (0) | 0/3 (0) | 0/3 (0) |
| reference (3) | 0/3 (0) | 0/3 (0) | 0/3 (0) | 0/3 (0) |
| sequence (3) | 0/3 (0) | 0/3 (0) | 0/3 (0) | 0/3 (0) |
| spatial_relations (3) | 0/3 (0) | 0/3 (0) | 0/3 (0) | 0/3 (0) |
| everyday_knowledge (3) | 0/3 (0) | 0/3 (0) | 0/3 (0) | 0/3 (0) |

★ = the categories I extended. The other five extension categories stayed out-of-vocabulary in
both experiments because I taught nothing for them; those zeros are by construction.

### Four-choice score vs. free continuation vs. coverage — three different measurements

The runner records all three for every case, and they can disagree:

| Case | Four-choice score | Free continuation (temp 0.8) | What the disagreement shows |
|---|---|---|---|
| starter `lang_07` "the report about the surgeon explains the" | 1 — `patient` ranked above traffic/delivery/fruit | `treatment in detail .` | The score only ranks the four given words; the model's own favourite (`treatment`, also a health-domain word) was not among them. Passing the score ≠ producing the answer. |
| extension `lang_30` "the opposite of noisy is" | 1 — `quiet` at 0.420 | `smooth .` | Sampling at temperature 0.8 does not always pick the top word; the score is deterministic, the continuation is not. |
| extension `lang_33` "…the door is" | 0 — `open` 0.010 vs `closed` 0.009 | `long .` | Both fail, differently: the four choices are all near zero and the sampled word is none of them. The model does not treat this as a place for a door state at all. |
| starter `lang_43` "water freezes into" | unscorable (`water`, `freezes`, `into` unknown) | `the new program focused on data and security .` | Coverage is a property of the vocabulary, not the weights: with every prompt word mapped to `<UNK>`, the model just emits a frequent template. No amount of training on the starter corpus changes this. |

Every case's continuation is in the linked `eval_results.json` files; the tables below quote
the ones that matter for the failures.

### Why I chose these three categories, and what I added

I chose **opposites** and **categories_and_analogies** because each has a short, repeatable
sentence frame that a tiny template-learner can plausibly pick up from a few hundred varied
examples — the same way it learned "the team discussed the X at the Y". I added **negation**
because it is a genuinely harder, three-sentence pattern where the answer must be *copied from
earlier context* rather than predicted from the local frame; I expected it to partly fail and
wanted a real failure to analyse.

[build_extension_corpus.py](build_extension_corpus.py) writes three files
(3,116 unique passages, 251 new word types; vocabulary went from 136 to 387):

| File | Lines | What it teaches | Example lines |
|---|---|---|---|
| [opposites.txt](corpus/extension/opposites.txt) | 806 | 30 antonym pairs in both directions, in up to 8 wordings (fewer for the tested pairs and open/closed; see below) | `the opposite of tall is short .` · `silent means not loud .` · `the room was big but now it is small .` |
| [categories.txt](corpus/extension/categories.txt) | 857 | 63 members of 10 categories; 10 young→adult animal pairs; two-sentence analogies | `a crow is a bird .a maple is a tree .` · `cotton belongs to the fabric group .` · `a young cat is called a kitten .` |
| [negation.txt](corpus/extension/negation.txt) | 1,453 | "X is not A . it is B . X is B" with 10 objects × 9 colours / 10 state pairs; "NAME did not V A . PRO V-ed B . NAME V-ed B" with 10 names × 8 verbs × 14 items; plain "NAME V-ed ITEM ." statements | `the gate is not white , the gate is green .` · `eli did not buy cake .he bought soup .eli bought soup .` · `ava ate cheese .` |

Two implementation notes. (1) The notebook loader splits text into passages at every `". "`,
which would separate the sentences of a negation or analogy example. I therefore wrote
multi-sentence lines without a space after the period (`green .it is white .`); the word tokenizer
produces the identical token sequence `green . it is white`, and the run's
[corpus.txt](experiments/extension/run/corpus.txt) shows them stored as single passages
(e.g. `a hammer is a tool . a trout is a fish .`). (2) I generated the lines with a script
rather than by hand so that every inclusion and exclusion rule is visible and reproducible.

### Keeping the tests out of training

**What the notebook does.**

- It removed **160** generated classroom sentences containing the 16 reserved test prefixes
  before splitting or building the vocabulary — recorded in `eval_separation.json`
  ([starter](experiments/starter/run/eval_separation.json) ·
  [extension](experiments/extension/run/eval_separation.json), all 16 case IDs listed).
- `reject_eval_leakage()` scanned every imported file and every final passage for exact test
  prompts; the import would have raised an error if any matched. It did not.
- `validate_corpus_location()` confirms `CORPUS_FOLDER` is `corpus/`, not the project root or
  `evals/`.
- The vocabulary is built from `train_docs` only; eval text never enters it. The runner only
  receives the prompt; the answer key is used afterwards, in the scoring loop.

**What my generator does.** The notebook's check only catches exact prompts, so
[build_extension_corpus.py](build_extension_corpus.py) applies four stricter rules (listed in its
docstring):

1. No teaching line contains any complete sentence of a test prompt or its answer, e.g.
   `a robin is a bird .`, `it is blue .`, `she bought milk .` or `the opposite of hot is cold .`.
2. **Opposites:** the three tested pairs never appear with "the opposite of", in either direction.
   They appear only as facts such as `hot and cold are opposites .` and `hot means not cold .`.
3. **Negation:**
   - The test subjects `box`, `door` and `ava` never appear in a negation sentence.
   - No line with "not" contains a tested pair (red/blue, open/closed, tea/milk).
   - No line contains two or more details of one test story.
   - The three test subjects still appear elsewhere so they are in the vocabulary, e.g.
     `one box is heavy and the other box is light .` and `ava ate cheese .`.
   - open/closed is a real antonym pair, so it is stated only as a plain fact
     (`open and closed are opposites .`), never with "not" or "it is".
4. **Categories:** facts such as "a salmon is a fish" are taught only in other wordings
   (`the salmon is a kind of fish .`). The assignment's own example allows this: its corpus
   teaches surgeon → patient in other sentences while withholding the test's wording.

**Independent check.** [check_leakage.py](check_leakage.py) reads the eval file and scans everything
a run trained on: its `corpus.txt`, i.e. the classroom sentences plus my files. It looks for:

- (A) eval prompts
- (B) any sentence of a test prompt or answer
- (C) negation-story details and tested pairs
- (D) tested opposite pairs with "opposite of"

Both runs: **0 matching lines** ([starter report](experiments/starter/leakage_check.txt) ·
[extension report](experiments/extension/leakage_check.txt)). It also reports the closest teaching
line for each extended case. What they share is frame words only:

| Case | Longest word-for-word run shared with prompt + answer | Example teaching line |
|---|---|---|
| lang_28–30 opposites | `the opposite of` (3 of 6 tokens) | `big is the opposite of small .` |
| lang_31 box, red → blue | `is not red . it is` (6 of 14) | `the ball is not red . it is black . the ball is black .` |
| lang_32 ava, tea → milk | `did not buy` (3 of 13) | `ben did not buy bread , he bought honey instead .` |
| lang_33 door, open → closed | `. it is` (3 of 14) | `the ball is not black . it is white . the ball is white .` |
| lang_46 robin / salmon | `is a bird . a` (5 of 11) | `a crow is a bird . a bicycle is a vehicle .` |
| lang_47 puppy / kitten | `grows into a` (3 of 13) | `a calf grows into a cow .` |
| lang_48 carrot / apple | `is a vegetable . an` (5 of 11) | `a broccoli is a vegetable . an eagle is a bird .` |

**The superseded first version.** My first extension run (27/48) banned only the tested object or
name. So some lines differed from a test by one word and still ended in its answer. There were
22 lines like `the box is not open .it is closed .the box is closed .` next to lang_33, and the
premise `a puppy grows into a dog .` appeared in 13 lines. `check_leakage.py` finds 200 matches
in that run's corpus. The assignment says to remove leaked material and rerun, so I tightened the
generator and trained a fresh model. Everything in this README comes from the rerun. The earlier
files remain only in git history
([commit f30a7bd](https://github.com/anthonybrites-cmyk/custom-llm-class4/tree/f30a7bd/experiments/extension)).
The comparison taught me something: lang_33 fell from ✔ 0.957 to ✘ 0.009. The earlier pass came
from near-duplicate lines, not from the model reading the negation.

Ordinary vocabulary overlaps by necessity. The words `robin`, `salmon`, `carrot`, `box`, `door`,
`ava`, `tea`, `milk` and the distractors (`fabric`, `warm`, `loud`, …) must exist in the
vocabulary for a case to be scorable at all. They never appear in the test sentences themselves.

### What the extension actually changed — with the failures

Per-case results and free continuations for all 48 cases are in the linked `eval_results.json`
files. Every extended case, plus the transfer cases whose score changed between the two trained
models:

| Case | Prompt (answer) | Starter trained | Extension trained | What happened |
|---|---|---|---|---|
| lang_28 | the opposite of hot is (cold) | unscorable | ✔ cold **0.898** | hot/cold never appears with "the opposite of", so this is real transfer. The model learned the frame from 27 other pairs and the hot↔cold link from other wordings (`hot and cold are opposites .`), and combined them. `warm`, a tempting distractor, got 0.000. |
| lang_29 | the opposite of empty is (full) | unscorable | ✔ full 0.888 | same |
| lang_30 | the opposite of noisy is (quiet) | unscorable | ✔ quiet 0.420 | Correct but the least confident of the three. The **free continuation was `smooth`**: sampling at 0.8 and the argmax score can disagree. |
| lang_46 | a robin is a bird . a salmon is a (fish) | unscorable | ✔ fish 0.975 | Neither `a robin is a bird .` nor `a salmon is a fish .` appears in the corpus. It combined `the salmon is a kind of fish` with the two-sentence frame learned from other members. |
| lang_48 | a carrot is a vegetable . an apple is a (fruit) | unscorable | ✔ fruit 0.968 | Same mechanism; the starter corpus also pairs apple with fruit heavily. |
| lang_47 | a puppy grows into a dog . a kitten grows into a (cat) | unscorable | ✘ **horse 0.292**, cat 0.025 | **Failure.** kitten→cat is taught (`a young cat is called a kitten .`, `the kitten will become a cat one day .`) but never in the "grows into" frame. That frame would be the answer sentence. The model completes "grows into a ___" with a frequent adult animal and ignores the subject. Free continuation: `horse`. It tracks the *frame*, not the *subject*. |
| lang_33 | the door is not open . it is closed . the door is (closed) | unscorable | ✘ open 0.010, closed 0.009 | **Failure.** `door` never appears in a negation sentence, and open/closed never appears with "not". With those removed, the model cannot choose: all four choices are near zero. The superseded version passed this case only because of near-duplicate lines. |
| lang_31 | the box is not red . it is blue . the box is (blue) | unscorable | ✘ green 0.096, red 0.082, yellow 0.071, blue 0.023 | **Failure.** It guesses among colours, and `blue`, the word it should copy from six tokens back, gets the lowest probability. |
| lang_32 | ava did not buy tea . she bought milk . ava bought (milk) | unscorable | ✘ rice 0.059, bread 0.048, milk 0.017, tea 0.016 | **Failure.** It picks a generic food rather than either item from the story. Free continuation: `honey`. |
| lang_18 / 22 / 23 | e.g. yesterday our office discussed the platform and the (update) | ✘ | ✔ | Transfer went 4/8 → 7/8, but this is noise, not learning. On six of the eight transfer cases every choice gets ≤ 0.001 probability because the model is off-template. The exceptions are lang_17 (`health` 0.163 starter, 0.269 extension) and lang_20 (`fruit` 0.007). |

**Was the change vocabulary coverage, learned patterns, or both?** Both, and they can be
separated. Coverage rose 24 → 33 purely because the words now exist: the *untrained* extension
model already shows 33/48 scorable. Of the 9 newly scorable cases, the trained model got 5,
versus 1 untrained. That difference is the learned patterns: the opposites frame (3/3, transferred
to pairs it never saw in that frame) and the category frame (2/3). Negation shows the limit. The
vocabulary is present and the three-sentence template appears in free samples, yet 0 of 3 cases
pass, because the pattern requires copying from context, not completing a frame.

**Rerunning the evals on the saved model.** The trained models are committed in this repo —
[experiments/starter/run/model.pt](experiments/starter/run/model.pt) and
[experiments/extension/run/model.pt](experiments/extension/run/model.pt) (each also inside its
`results.zip`). Each `model.pt` holds the full network state, the model config and the saved
vocabulary; `run_evals.load_model()` rebuilds the nanoGPT from it with no download, API key or
pretrained weights. Dependencies are just `requirements.txt` (torch, numpy). When I tested the
command below, it reproduced the notebook's final result exactly: same 28/48, same
probabilities, same free continuations, same model hash.

```bash
.venv/bin/python run_evals.py --model experiments/extension/run/model.pt --output results/rerun --stage rerun
```

(Use `experiments/starter/run/model.pt` to reproduce 20/48. The runner refuses a non-empty
output directory so earlier results are never overwritten.)

## Chat interface

The interface is the supplied [chat.py](chat.py): a terminal loop that loads `model.pt` and
its saved vocabulary, tokenizes each prompt, samples up to 24 tokens at temperature 0.8, and
reports unknown prompt words. Each prompt starts fresh (no conversation memory); prompts longer
than 48 tokens are truncated to the most recent 48; nothing is retrained or added to the corpus.

To chat yourself (type a prompt at `You:`, `/quit` to exit):

```bash
.venv/bin/python chat.py --model experiments/extension/run/model.pt --transcript results/my_chat.json
```

**Model used:** extension run `20260923T011020_029184Z`, 3,000 steps, `model_sha256`
`88237999e17b277a…`. This is the same value recorded in the final eval summary and in the
chat transcript.

Four real interactions ([transcript](experiments/extension/chat_transcript.json)). I ran the
unchanged `chat.py` in a terminal. The prompts were typed at its `You:` prompt by
[chat_session.exp](chat_session.exp), a four-line `expect` script, so the exact session can be
replayed with `expect chat_session.exp`. The screenshot is of that terminal:

![chat screenshot](experiments/extension/chat_screenshot.png)

| # | Prompt | Model | Verdict |
|---|---|---|---|
| 1 | `the doctor` | `and the birch are both trees .` | **Failure.** It stitched the categories frame ("the X and the Y are both Zs") onto a starter noun. Mixing two corpora produced a cross-template sentence that means nothing. |
| 2 | `the opposite of tall is` | `short .` | Correct: tall/short is a taught pair in a taught frame. |
| 3 | `my favorite umbrella` | `.` (unknown words: favorite, my, umbrella) | **Limitation.** All three words are `<UNK>`, so the prompt carried no information; the model emitted a lone period and stopped. |
| 4 | `the cup is not red . it is white . the cup is` | `grey .` | **Failure**, consistent with the negation evals. This exact story is not in training (the corpus has cup red → pink), so the model had to copy `white` from the prompt, and it didn't. |

The notebook's own chat cell (section 10) also ran once during Run All, with the same model in
memory: `the customer` → `reviewed the item after checking the price .`, saved in the run folder's
[chat_transcript.json](experiments/extension/run/chat_transcript.json). The interface prints its
own labels: "Tiny language model: short continuations, not a general assistant", the 48-token
context, that each prompt starts fresh, and any unknown words (visible in turn 3 above).

**Observed chat limitation:** the model has no notion of a question or a request. Every prompt
is treated as the beginning of a sentence to continue, and any word outside the 387-word
vocabulary is silently replaced by `<UNK>` before the model sees it. So a prompt made only of
unknown words (turn 3) carries no information at all, and the reply is effectively unconditional.

## One limitation and one next experiment

**Limitation.** The model reproduces sentence *frames*, not relations. Negation (0/3 evals, 1/1
chat failure) and the kitten→horse analogy show it completing a frequent filler for a frame
instead of reading the specific words earlier in the context. Its own halfway sample shows this
in one line: `the shirt is not green . it is grey . the shirt is yellow .` is fluent, has the exact
negation shape, and contradicts itself. The superseded run makes the same point from the other
side. Near-duplicate training lines made a negation case pass, and the pass disappeared when they
were removed.

**Next experiment.** Change one thing: keep the corpus and settings and raise `N_LAYER` from 2
to 4, then rerun only the three negation evals plus the ten transfer/starter cases as a
regression check. The hypothesis is that copying a word from six tokens back needs more
attention depth than two blocks provide. If negation does not improve, the alternative is
data-side. Negation stories are only about a fifth of the unique passages (1,453 of 7,708).
Doubling them would test whether the model lacks capacity or just examples.

## Embedding viewer

`checkpoint.json` from either run loads in [embedding-viewer.html](embedding-viewer.html)
(open locally, "Open your checkpoint"). It contains initial and final `wte` embeddings only —
not the full network; `model.pt` is the inference model. The viewer's 3D map is a PCA projection
that discards most of the 64 dimensions; the cosine neighbours quoted above use all 64.
