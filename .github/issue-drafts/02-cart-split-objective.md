## Lesson metadata

- **Notebook:** [`notebooks/02-cart-split-objective.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/02-cart-split-objective.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Lesson 01's regions, leaves, and mean predictions
- **Depends on:** Lesson 01

## Primary objective

Derive and implement how a CART regression stump enumerates candidate thresholds and chooses
the split that minimizes the total squared error in its child leaves.

## Tutor must explain first

Before assigning implementation, explain:

- why useful thresholds lie between sorted distinct feature values;
- how a candidate threshold creates left and right targets;
- why the mean minimizes squared error inside a fixed leaf;
- why child losses must be added to evaluate the split; and
- the distinction between leaf prediction, leaf loss, split loss, and loss reduction.

Keep terminology consistent: use squared-error/RSS language and explicitly note equivalent
terms only once.

## Real-data intuition and required visual

Reuse a tiny deterministic California Housing slice with one feature. Display sorted feature
values, midpoint candidates, child means, left loss, right loss, and total loss in a compact
table.

Plot total split loss against threshold and highlight the selected threshold. Pair it with the
resulting two-step prediction plot so the numerical objective stays connected to geometry.

## Derivation target

Derive, step by step in LaTeX:

1. the constant prediction minimizing squared error in a fixed region;
2. the summed left-plus-right objective for one candidate threshold; and
3. the argmin that selects the best candidate.

Map every symbol to the displayed table and eventual Python variables. Do not introduce
classification impurity, recursive growth, Hessians, or approximate splits.

## Learner-owned work

With the formulas and candidate-generation rule already explained, have the learner implement
typed helpers in `tree.py` for:

- generating midpoint thresholds from one feature;
- computing squared error around a leaf mean; and
- evaluating/selecting a one-dimensional split.

It is reasonable to scaffold signatures or one loop, but the learner should own the loss
calculation and selection logic.

## Files touched

- `notebooks/02-cart-split-objective.ipynb`
- `src/boosted_trees_course/tree.py`
- optionally `src/boosted_trees_course/visualization.py`

## Notebook checks and mastery questions

Use visible, hand-computable values to assert:

- candidate count matches the number of distinct adjacent gaps;
- no candidate produces an empty child;
- the selected loss equals the minimum displayed candidate loss; and
- constant targets produce zero loss regardless of a valid split.

Ask the learner to predict how duplicate feature values and an extreme outlier affect the
candidate set and objective.

## Deliverables

- [ ] Concise CART regression split derivation
- [ ] Candidate-loss table and threshold-loss plot
- [ ] Typed one-feature split helpers
- [ ] Explained notebook assertions on the tiny real slice
- [ ] Learner explanation of why the chosen threshold wins

## Stopping condition

Stop when the learner can calculate and implement the best split for one feature at one node.
Do not recurse or build a full estimator.

## Optional extension

Compare total squared error with mean squared error only to explain when child-size weighting
matters; keep one canonical objective in the implementation.
