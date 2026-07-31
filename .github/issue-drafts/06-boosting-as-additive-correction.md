## Lesson metadata

- **Notebook:** [`notebooks/06-boosting-as-additive-correction.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/06-boosting-as-additive-correction.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Custom CART prediction and squared-error behavior
- **Depends on:** Lessons 01–05

## Primary objective

Develop the core intuition that boosting builds an additive prediction function by repeatedly
fitting a small tree to the current model's desired corrections.

## Tutor must explain first

Before assigning a boosting loop, explain:

- why an initial constant prediction is useful;
- residuals as target minus current prediction under squared error;
- why a new tree predicts a correction rather than the original target;
- how the learning rate scales each correction; and
- how frozen earlier trees plus a new correction differ from jointly updating neural-network
  parameters.

Use the learner's gradient-descent background, while making clear that trees define update
functions rather than parameter gradients in one fixed network.

## Real-data intuition and required visual

Use a tiny one-feature California Housing slice. Starting from the target mean, manually
build or fit a sequence of roughly three shallow custom trees to residuals.

Create a staged figure showing, for every stage:

- current predictions;
- residuals/corrections requested by the rows;
- the new tree's step-shaped correction; and
- the updated ensemble prediction.

A small reproducible animation is optional only if it is clearer than static panels.

## Derivation target

Write the additive-model update and squared-error residual relationship in concise LaTeX.
Map the stage index, learning rate, correction tree, target, and current prediction directly to
notebook variables. Do not introduce general loss gradients or Hessians yet.

## Learner-owned work

After the process is fully demonstrated once, have the learner implement a short notebook
loop that:

- initializes predictions;
- computes residuals;
- fits one shallow custom CART regressor to them;
- applies a learning-rate-scaled update; and
- records staged predictions and loss.

Keep this loop explicit; do not hide it in the final booster class yet.

## Files touched

- `notebooks/06-boosting-as-additive-correction.ipynb`
- existing `src/boosted_trees_course/tree.py`
- `src/boosted_trees_course/visualization.py` for staged plots if reusable

## Notebook checks and mastery questions

Use checks such as:

- the initial prediction is constant;
- one residual exists per training row;
- the new prediction equals the old prediction plus the scaled tree output; and
- staged squared loss decreases on the tiny training slice for the selected safe settings.

Ask the learner to predict how a smaller learning rate changes one stage and why many shallow
trees can express more than one stump.

## Deliverables

- [ ] Three-stage real-data boosting walkthrough
- [ ] Static staged-correction visual (or justified small animation)
- [ ] Explicit notebook boosting loop
- [ ] Staged-loss record and explained assertions
- [ ] Learner explanation of boosting in gradient-descent language

## Stopping condition

Stop when the learner can narrate one complete boosting stage and reconstruct the ensemble
prediction as an initial value plus scaled tree outputs. Leave generalized gradients and the
reusable class for later lessons.

## Optional extension

Compare two learning rates for the same three correction trees and predict the resulting
under/over-correction before plotting.
