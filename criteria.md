# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1.

<!-- Your criterion. It must name a number. -->
Overall accuracy ≥ 0.75 on the held-out test split, on each of the 3 seeds, and at least 15 points above the majority-class share.



**Why this target:**
With 4 different labels, random guessing gets 25%. The "15 points above majority" part keep the target meaningful for whichever label ends up biggest

 
---

## 2.
<!-- Your criterion. -->
Macro F1 ≥ 0.65, and every label's F1 ≥ 0.50, on each of the 3 seeds.



**Why this target:**
A strong news score will not a weak off-topic score. The per-label floor is loose because the smallest label will have a small amount of test, so one miss will move F1 by about 0.15 to 0.2

---

## 3.

<!-- Your criterion. -->
3. Every label is at least 15% of my labelled data, and each has at least 5 posts in the test split.



**Why this target:**
Below the 15% value, a label has 4 or fewer test posts and its F1 in #2 become noise. 

---

## 4.
<!-- Your criterion. -->
At least 80% agreement with the staff labels in agreement.py

**Why this target:**
This is the only check on whether my labels are consistent. If I can't match a writtene rule 80% of the time, low model scores could be the label's fault, not the model's fault.


---

## 5.
<!-- Your criterion. -->
Predictions with confidence >= 0.9 are correct at least 12 points more often than predictions below 0.9, if each bucket has at least 5 posts. Otherwise, it will be reported as not measurable for that seed.


**Why this target:**
Sarcastic-humor and news should produce unsure predictions.


---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
