## Lesson metadata

- **Notebook:** [`notebooks/09-second-order-objective-and-split-gain.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/09-second-order-objective-and-split-gain.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** General loss gradients, raw predictions, and CART split search
- **Depends on:** Lessons 01–08

## Primary objective

Develop the second-order intuition behind XGBoost-style leaf values and split gain by
aggregating per-row gradients and Hessians inside candidate leaves.

## Tutor must explain first

Before assigning calculations, explain:

- gradient as local slope and Hessian as local curvature with respect to a prediction;
- why a second-order Taylor approximation gives a local quadratic objective;
- why rows in one leaf share one added weight;
- how gradient/Hessian sums summarize the leaf's local objective;
- how L2 leaf regularization shrinks weights and stabilizes small-curvature leaves; and
- why a split is useful only when two child optima beat the parent after penalties.

Refresh only the derivative/Taylor facts needed for these steps.

## Real-data intuition and required visual

Use a tiny California Housing slice and visible current predictions. Compute per-row gradient
and Hessian values for squared error and, in a second compact example, binary logistic loss.
Group rows under one candidate split and display parent/left/right aggregate quantities.

Create a Matplotlib visual showing how regularization changes leaf weights and split gain.
A small gain heatmap over candidate thresholds and regularization strengths is encouraged.

## Derivation target

Derive in a single coherent sequence:

1. the per-stage second-order Taylor approximation;
2. the local objective for a constant leaf weight;
3. the regularized optimal leaf weight;
4. the optimized leaf score; and
5. parent-versus-children split gain with a split penalty.

Map each aggregate symbol directly to arrays and scalar variables. Do not prove Taylor's
theorem, implement tree recursion, or introduce histogram approximations in this lesson.

## Learner-owned work

After the derivation and one worked parent/child example, have the learner implement typed
helpers for:

- aggregating gradient/Hessian values;
- computing a regularized leaf weight; and
- computing split gain from parent and child aggregates.

The learner should reproduce one hand-computed gain with code and investigate how penalties
can reject a split.

## Files touched

- `notebooks/09-second-order-objective-and-split-gain.ipynb`
- `src/boosted_trees_course/losses.py` for Hessian support
- `src/boosted_trees_course/xgb_style.py` for leaf-weight and gain helpers
- optionally `src/boosted_trees_course/visualization.py`

## Notebook checks and mastery questions

Use explained checks such as:

- hand-computed and helper-produced leaf weights/gains agree;
- increasing L2 regularization shrinks the magnitude of a fixed leaf weight;
- a sufficiently large split penalty rejects a marginal split;
- squared-error Hessians have the expected simple structure; and
- logistic Hessians stay nonnegative for the visible finite scores.

Ask the learner why Hessian information changes confidence/step scaling and why gain compares
optimized objectives rather than raw target variance.

## Deliverables

- [ ] Concise Taylor-to-gain derivation
- [ ] Hand-computable parent/child real-data table
- [ ] Typed gradient/Hessian aggregate, leaf-weight, and gain helpers
- [ ] Regularization/gain visual
- [ ] Learner explanation of every term's behavioral effect

## Stopping condition

Stop when the learner can derive, calculate, implement, and interpret one regularized leaf
weight and one split gain. Leave recursive second-order tree construction for Lesson 10.

## Optional extension

Show the limiting relationship between the squared-error second-order leaf update and a
residual mean under simplified settings, without introducing another implementation path.
