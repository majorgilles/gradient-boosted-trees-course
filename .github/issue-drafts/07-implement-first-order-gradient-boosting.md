## Lesson metadata

- **Notebook:** [`notebooks/07-implement-first-order-gradient-boosting.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/07-implement-first-order-gradient-boosting.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Explicit staged residual-fitting loop and typed CART regressor
- **Depends on:** Lessons 01–06

## Primary objective

Turn the explicit squared-error boosting loop into a typed, reusable first-order
gradient-boosting regressor and connect each method to stage-wise optimization over
predictions.

## Tutor must explain first

Before refactoring, explain:

- the state an additive estimator must store;
- why the initial prediction and each fitted tree remain frozen;
- the distinction among estimators, stages, depth, and leaves;
- how `fit`, `predict`, and staged predictions compose; and
- why learning rate and number of estimators trade step size against iteration count.

Give an intuition-first account of gradient descent in function/prediction space without a
formal functional-analysis detour.

## Real-data intuition and required visual

Use a fixed California Housing subset that the custom exhaustive tree can handle. Plot:

1. training and validation loss across boosting stages;
2. staged one-feature prediction curves for selected early and late stages; and
3. optionally a learning-rate/estimator-count comparison with approximately matched total
   update opportunity.

## Derivation target

Start from the additive stage update and show why the negative squared-error gradient equals
the familiar residual. Map the loss derivative with respect to each current prediction to the
training target passed into the next tree.

Do not generalize the software to multiple losses yet; Lesson 08 owns that refactor. Do not
use XGBoost, LightGBM, or scikit-learn boosting internally.

## Learner-owned work

Have the learner implement the meaningful pieces of a typed
`GradientBoostingRegressor`, including:

- constructor/state for learning rate, number of estimators, and tree settings;
- initial constant fit;
- sequential residual-tree fitting;
- batch prediction as an additive sum; and
- staged prediction or staged loss inspection.

Build directly on the custom CART regressor. Use partial signatures only after the expected
state transitions are explained.

## Files touched

- `notebooks/07-implement-first-order-gradient-boosting.ipynb`
- `src/boosted_trees_course/boosting.py`
- existing `src/boosted_trees_course/tree.py`
- optionally `src/boosted_trees_course/visualization.py`

## Notebook checks and mastery questions

Use notebook checks such as:

- zero fitted trees produce the initial constant prediction;
- the stored tree count matches completed stages;
- batch prediction equals the explicitly reconstructed additive sum;
- staged outputs have consistent shapes and finite values; and
- the selected conservative setup reduces training loss across observed stages.

Ask the learner why training loss need not guarantee validation improvement and what would
happen if old trees were retrained after every stage.

## Deliverables

- [ ] Typed first-order gradient-boosting regressor
- [ ] Additive prediction reconstruction check
- [ ] Staged training/validation loss plot
- [ ] Learning-rate interpretation
- [ ] Learner explanation of function-space update intuition

## Stopping condition

Stop when the custom first-order regressor fits, predicts, exposes staged behavior, and the
learner can map every stored component to the additive model. Leave generalized losses for
Lesson 08.

## Optional extension

Add an early-stopping *observation* based on a validation curve, without building a
production callback or mutating the core training contract.
