## Lesson metadata

- **Notebook:** [`notebooks/10-implement-xgboost-style-learner.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/10-implement-xgboost-style-learner.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Typed CART recursion plus second-order leaf-weight and gain helpers
- **Depends on:** Lessons 01–09

## Primary objective

Compose the second-order objective helpers into a compact typed tree learner and additive
booster that captures the essential educational mechanics behind XGBoost.

## Tutor must explain first

Before implementation, compare the custom CART and second-order tree-building loops:

- CART evaluates target-variance reduction and predicts target means;
- the second-order tree evaluates gradient/Hessian gain and predicts regularized update
  weights;
- both recursively partition rows with thresholds and stopping rules;
- boosting recomputes derivatives at every stage while old trees remain frozen; and
- depth, minimum gain, L2 leaf regularization, split penalty, learning rate, and estimator
  count act at different points.

Clarify raw score versus transformed prediction for binary loss.

## Real-data intuition and required visual

Develop on a small California Housing subset, then run controlled regression behavior checks
on a larger manageable subset. Use the high-price target for one compact binary raw-score
walkthrough if it remains within the lesson timebox.

Create visuals for:

1. selected candidate gains at one node;
2. staged loss for the compact ensemble; and
3. the effect of leaf or split regularization on tree size/update magnitude.

## Derivation target

No new major derivation. Build an implementation map from Lesson 09's formulas to:
objective derivatives, split enumeration, node recursion, leaf construction, stage update,
and transformed prediction. Keep formula repetition minimal.

Do not add histogram binning, missing-value default directions, row/column sampling,
parallelism, or a scikit-learn-compatible estimator surface.

## Learner-owned work

Have the learner implement the meaningful missing parts of a minimal typed learner, such as:

- second-order best-split search using gradient/Hessian aggregates;
- recursive second-order node growth;
- one-tree update prediction;
- additive stage fitting with derivative recomputation; and
- raw and transformed prediction paths where applicable.

Reuse clear node/traversal pieces from `tree.py` when useful, but do not force inheritance if
it obscures the two objectives.

## Files touched

- `notebooks/10-implement-xgboost-style-learner.ipynb`
- `src/boosted_trees_course/xgb_style.py`
- `src/boosted_trees_course/losses.py`
- selected reusable node/visual helpers from `tree.py` or `visualization.py`

## Notebook checks and mastery questions

Use checks such as:

- zero-tree output equals the objective's base raw prediction;
- each fitted leaf value matches the Lesson 09 helper for its rows;
- predictions equal base score plus scaled tree updates;
- stronger leaf regularization shrinks selected update magnitudes;
- stronger split penalty does not increase the number of accepted splits in the controlled
  example; and
- staged outputs remain finite and improve the selected early-stage objective.

Ask the learner to contrast one CART node and one XGBoost-style node using the exact arrays
each consumes.

## Deliverables

- [ ] Typed compact second-order tree and booster
- [ ] Node-level gain inspection
- [ ] Staged-loss and regularization visuals
- [ ] Explained notebook assertions
- [ ] Learner comparison of CART, first-order, and second-order training loops

## Stopping condition

Stop when the compact learner fits and predicts on controlled real-data subsets and the
learner can trace one row from derivatives through splits, leaf weight, scaled update, and
ensemble prediction. Do not optimize the implementation.

## Optional extension

Run the same tree structure with squared-error and logistic objectives to isolate how
derivatives—not partition/traversal mechanics—change.
