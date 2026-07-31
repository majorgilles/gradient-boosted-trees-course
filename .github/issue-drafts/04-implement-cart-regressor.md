## Lesson metadata

- **Notebook:** [`notebooks/04-implement-cart-regressor.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/04-implement-cart-regressor.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** One-node split search, typed node structures, recursion, and stopping rules
- **Depends on:** Lessons 01–03

## Primary objective

Complete a small, typed CART-style regression estimator from first principles by composing
the split and node machinery developed in prior lessons.

## Tutor must explain first

Before implementation, review the complete data flow:

1. validate/store training arrays;
2. evaluate every allowed feature and threshold at a node;
3. reject invalid or insufficient-gain splits;
4. recurse into child row subsets or create a mean-valued leaf; and
5. traverse the fitted node structure for prediction.

Explain array shape conventions, fitted state, deterministic tie handling, and why the
educational algorithm is intentionally exhaustive and slow.

## Real-data intuition and required visual

Use a small multifeature California Housing slice while developing the estimator. After it
fits, draw its shallow tree and show predicted step regions for a one- or two-feature view.
Keep node sample counts and leaf values inspectable.

## Derivation target

Consolidate prior formulas into an implementation map rather than adding new mathematics:
node loss, candidate split loss/reduction, stopping decisions, and leaf prediction. Explicitly
map each term to a helper, field, or local variable.

Do not derive classification trees, pruning, vectorized histogram splits, or scikit-learn
estimator compatibility.

## Learner-owned work

The learner should implement the meaningful missing pieces of a minimal typed estimator,
such as:

- multifeature best-split search;
- recursive node growth;
- single-row tree traversal;
- batch `predict`; and
- basic fitted/input validation.

Build on existing Lesson 02–03 code instead of replacing it with an external tree. Scaffold
signatures and one representative branch only if needed; explain every gap first.

## Files touched

- `notebooks/04-implement-cart-regressor.ipynb`
- `src/boosted_trees_course/tree.py`
- optionally `src/boosted_trees_course/visualization.py`

## Notebook checks and mastery questions

Use a tiny real slice to assert behavior such as:

- `predict` returns one finite value per row;
- a depth-zero model predicts the training-target mean;
- maximum observed depth does not exceed `max_depth`;
- every leaf respects the configured minimum sample count; and
- repeated fits with the same data and settings are deterministic.

Ask the learner to explain the call path for one prediction and the computational bottleneck
in exhaustive split search.

## Deliverables

- [ ] Typed educational CART regressor in `tree.py`
- [ ] Shallow-tree fit on a transparent California slice
- [ ] Tree or region visualization
- [ ] Explained behavioral assertions
- [ ] Learner walkthrough of fit and prediction control flow

## Stopping condition

Stop when the custom estimator can fit and predict deterministically on the selected real-data
slice and the learner can explain every major method. Performance optimization is not part of
this issue.

## Optional extension

Add a typed method that yields leaf decisions for one row to improve inspectability; avoid
building a general model-explanation framework.
