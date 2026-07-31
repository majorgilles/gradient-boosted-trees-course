## Lesson metadata

- **Notebook:** [`notebooks/08-general-losses-and-binary-classification.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/08-general-losses-and-binary-classification.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** First-order squared-error booster and derivative basics
- **Depends on:** Lessons 01–07

## Primary objective

Generalize “fit residuals” into “fit negative loss gradients” and use a focused binary
logistic-loss extension to show why gradient boosting is not fundamentally tied to ordinary
residuals.

## Tutor must explain first

Before asking for a loss abstraction, explain:

- loss as a function of the target and current raw prediction;
- derivative with respect to the current prediction, not model parameters;
- negative gradient as the desired first-order correction;
- logits versus probabilities and why boosting updates raw scores; and
- how sigmoid and binary log loss connect target labels to gradient direction.

Create the high-price target explicitly as a pedagogical label. When using a held-out split,
compute its threshold from training targets only.

## Real-data intuition and required visual

Use a tiny California Housing slice containing rows on both sides of the training-derived
high-price threshold.

Plot current probabilities and negative gradients for positive and negative rows at several
raw-score values. Include a small table showing how confidently wrong examples request
larger corrections than confidently correct examples.

## Derivation target

Derive only what is needed:

1. squared-error gradient and its residual equivalence;
2. sigmoid transformation from raw score to probability;
3. binary-log-loss gradient with respect to the raw score.

Map every symbol to a typed loss method and notebook array. Save the Hessian and Taylor
approximation for Lesson 09. Multiclass softmax is out of scope.

## Learner-owned work

After both losses are explained, have the learner:

- define a small typed first-order loss protocol/abstraction;
- implement squared-error and binary-log-loss values/gradients;
- refactor the first-order booster to request negative gradients from the loss; and
- run a compact binary boosting experiment using the derived real-data label.

Scaffold stable numerical helpers if needed, but explain clipping/stability before providing
them.

## Files touched

- `notebooks/08-general-losses-and-binary-classification.ipynb`
- `src/boosted_trees_course/losses.py`
- `src/boosted_trees_course/boosting.py`
- `src/boosted_trees_course/data.py` for leakage-safe label derivation

## Notebook checks and mastery questions

Use checks such as:

- probabilities stay within the expected range and preserve raw-score ordering;
- squared-error negative gradients match residuals on visible values;
- binary gradients point in the expected direction for positive/negative labels;
- the threshold is computed from training targets only; and
- selected first-order binary-loss values improve across early stages.

Ask the learner why fitting `y - probability` resembles residual fitting but occurs in raw
score space.

## Deliverables

- [ ] Typed first-order loss abstraction
- [ ] Squared-error and binary-log-loss implementations
- [ ] Leakage-safe high-price target helper
- [ ] Gradient-direction visual and explained assertions
- [ ] Focused binary boosting experiment and learner explanation

## Stopping condition

Stop when the learner can explain residuals as one special negative gradient and can trace a
binary label through raw score, probability, gradient, correction tree, and updated score.
Do not introduce Hessian-weighted gain yet.

## Optional extension

Numerically compare an analytic gradient with a small finite-difference estimate for a few
visible values; keep this a notebook sanity check, not a testing framework.
