# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

<!-- Your community, and what your classifier sorts posts into. Three or four
     sentences. -->

I am posting about the Formula 1 reddit page. My classifiers is news, humor, off-topic, and opinion. These classifiers cover the posts that mostly appear in the F1 reddit page.
---

** Candidate distinctions I saw while going through posts:
- Most news post cite a website or a source.
- Humorous post usually include the joke in the post if told by person in Formula 1
- There are a lot of images on the F1 page
## Label Taxonomy

<!-- Each label: a one-sentence definition and two real examples from your
     reading. Then your decision rule for the hardest boundary.

     The decision rule is worth a point on its own and it's the thing most
     people leave out. Every taxonomy has a hardest boundary. Name yours. -->

### `News`

**Definition:** providing updates news on the latest transfer, driver stories, and results around F1

**Example 1:**
>
[Autoracer] Ferrari unveils the fourth SF-26 package at Austin:


**Example 2:**
>
[Valtteri Bottas] 419.67km commute; Well arrived in Singapore this arvo.



### `humor`

**Definition:** attempting to make funny statements or make a joke about certain statistics of drivers;

**Example 1:**
>
Congratulations to Lance Stroll for Completing his 1st Full Race This Year


**Example 2:**
>
Oscar Piastri’s pre-f1 era tweets are so funny

### `opinion`

**Definition:** giving an opinion on an F1 that is deemed to cause a discussion

**Example 1:**
>
In my opinion the most down to earth and genuine F1 driver out there.


**Example 2:**
>
In my opinion, the SF71-H is the prettiest looking Ferrari since the F2004. Photo credit goes to u/Kerrentonsnow

### `off-topic`

**Definition:** posts that are not F1 news (current or historical), not a joke about F1, and not an opinion about F1, such as other motorsports, F1 drivers' lives outside their F1 careers (vacations, advice), or general questions and random topics.

**Example 1:**
>
Parents went on vacation and brought me back my first F1 merchandise!


**Example 2:**
>
Discussion thread. Where would you take a week long F1 vacation? (A little game to pass the time)



### The hardest boundary

**Which two labels:** news and humor
"Do they think I have a button that says, ‘George engine failure now’?": Toto Wolff slams idea of Mercedes favoring Kimi Antonelli over George Russell


**The decision rule I used every time:**
<!-- e.g. "If the post names a specific checkable fact, it's `analysis`, even
     if the tone is heated." -->
If the post has any newsworthy information, it is listed as news, even if there is sarcasm or humor in the tweet.


---

## The Dataset

<!-- Where you collected from, how you labelled, your counts, and three hard
     cases. -->

**Where the posts came from:**
The post came from the F1 reddit page.

**How I labelled them:** <!-- Cold first? Pre-labelled with AI and corrected?
Say so plainly — the disclosure is required, not penalised. -->
I labeled the first 24 cold, and then let AI prelabel the rest of them. I did review each of the post and correct some of them that I felt was inaccurate.

**Counts per label:**

| Label | Count | Share |
|---|---|---|
| news | 57 | 28.5% |
| humor | 57 | 28.5% |
| off-topic | 48 | 24% |
| opinion | 38 | 19% |
| **Total** |  | 100% |

**Three hard cases**

<!-- Any post that made you pause: what it was, which two labels it could have
     been, and what you chose. These are worth more than the easy 190. -->

**1.**
> *The post: "[The Race] Isack Hadjar to Sky following his sprint race retirement... 'except for Lewis, apparently!*
>
> *Could have been: news*
>
> *I chose humor, because: it was a driver making a joke about the story. Humor is making joke about statistics and that is what Isack did.*

**2.**
> *The post: Max Verstappen has now led more Grand Prix laps in Singapore than Antonio Giovinazzi (4).*
>
> *Could have been: news*
>
> *I chose humor, because: it was making a joke about driver's statistics*

**3.**
> *The post: Verstappen: F1 shouldn't hold sprint races on tracks where you can't pass",*
>
> *Could have been news:*
>
> *I chose opinion, because Max was giving an opinion on the topic of sprint races:*

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:** 
BASE_MODEL = "distilbert-base-uncased"

**Settings:** <!-- epochs, learning rate, batch size, seed -->
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
SEED = 42


**Anything I changed from the defaults, and why:**
I have left everything on default. I only added my 4 labels.

**Split sizes:** <!-- train / val / test, and per-label counts in the test
split. If a label had fewer than about 8 in test, say so — it explains a lot
of next unit's variance. -->

```python
from sklearn.model_selection import train_test_split


def split_dataset(df, seed, test_size=0.15, val_size=0.15):
    """
    Stratified 70/15/15. Returns (train, val, test) DataFrames.

    Stratifying falls back to a plain random split if any label is too rare to
    stratify on — that's a warning sign about the dataset, not about the split.
    """
    def _stratify_on(frame):
        counts = frame["label"].value_counts()
        return frame["label"] if counts.min() >= 2 else None

    rest, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=_stratify_on(df)
    )
    val_share = val_size / (1 - test_size)
    train, val = train_test_split(
        rest, test_size=val_share, random_state=seed, stratify=_stratify_on(rest)
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def describe_splits(train, val, test, labels):
    """A table of per-label counts per split, and a note on the thin ones."""
    import pandas as pd

    table = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).reindex(labels).fillna(0).astype(int)

    thin = [label for label in labels if table.loc[label, "test"] < 8]
    return table, thin


train_df, val_df, test_df = split_dataset(df, SEED)
table, thin = describe_splits(train_df, val_df, test_df, LABELS)

print(f"train {len(train_df)}  ·  val {len(val_df)}  ·  test {len(test_df)}\n")
print(table.to_string())

if thin:
    print()
    for label in thin:
        n = table.loc[label, "test"]
        print(f"[note] '{label}' has only {n} examples in the test split.")
    print()
    print("A metric computed on that few examples moves a lot between seeds.")
    print("Write that down now — next unit it's a diagnosis, not a surprise.")
```


**Per label test counts**

train 139  ·  val 31  ·  test 30

           train  val  test
label                      
news          40    9     8
humor         39    9     9
opinion       26    6     6
off-topic     34    7     7

---

## How I Used AI

<!-- Two specific moments — what you asked, what came back, what you changed.

     ⚠️ Plus disclosure of any pre-labelling. If you had a model pre-label a
     batch and then read and corrected every one, say that. It's an allowed
     workflow and disclosing it costs you nothing. Not disclosing it is the
     problem. -->

**Moment 1**

- *What I asked for:* to finish completing the rows on my labels.csv file
- *What came back:*  it returned the rows and labelled them "pre-labelled"
- *What I changed:* I reviewed the rows and made a couple of minor changes.

**Moment 2**

- *What I asked for:* advice on to run the takemeter.ipynb file
- *What came back:* it gave me advice on how to run the file 
- *What I changed:* I knew how to run the file now.

**Pre-labelling disclosure:**
I labeled the first 24 rows and marked them cold. I then allowed the AI to label the next 176 rows and marked them "pre-labeled". I then checked over what AI did and marked them "pre-labeled-checked" after completing them. I changed a couple from news to humor but left most of them with what the AI marked.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
