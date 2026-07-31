## Lesson metadata

- **Notebook:** [`notebooks/11-xgboost-lightgbm-production-bridge.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/11-xgboost-lightgbm-production-bridge.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Working compact second-order learner and exact split-gain intuition
- **Depends on:** Lessons 01–10

## Primary objective

Connect the inspectable custom learner to the main algorithmic and systems choices that make
XGBoost and LightGBM practical, without using their APIs or attempting to clone them.

## Tutor must explain first

Explain and clearly separate:

- exact threshold enumeration versus histogram/quantile-binned candidates;
- XGBoost's common depth-wise growth and configurable alternatives;
- LightGBM's leaf-wise best-first growth and why it can reduce loss quickly but overfit small
  data;
- how missing-value default directions and sparse-aware split evaluation differ from ordinary
  dense routing;
- row and feature sampling as regularization/efficiency tools; and
- where parallelism, cache-aware code, GPU, and distributed training improve systems
  performance without changing the core additive objective.

Mention LightGBM-specific techniques such as gradient-based sampling or feature bundling only
at an intuitive level and label defaults/version-sensitive claims carefully.

## Real-data intuition and required visual

Using derivatives from a California Housing subset, compare exact candidate evaluation with a
small number of deterministic feature bins. Show which threshold/gain each method chooses and
what information binning discards.

Create a second visual that starts from several candidate leaf gains and contrasts:

- splitting all nodes level by level; and
- splitting only the currently highest-gain leaf.

Use diagrams or Matplotlib panels; do not import XGBoost or LightGBM yet.

## Derivation target

Reuse the established gain formula. Define only the aggregate gradient/Hessian quantities
stored per histogram bin and how prefix sums recover left/right candidates. Avoid complexity
proofs, implementation-specific kernel details, or a complete missing-value derivation.

## Learner-owned work

Have the learner run one bounded exact-versus-binned split experiment and one paper simulation
of depth-wise versus leaf-wise growth. A tiny educational binning helper is acceptable, but a
new production tree builder is not.

The learner should write a comparison table covering objective mechanics, split search,
growth policy, regularization knobs, missing values, sampling, and major performance ideas.

## Files touched

- `notebooks/11-xgboost-lightgbm-production-bridge.ipynb`
- optionally `src/boosted_trees_course/visualization.py`
- optionally a very small inspection helper in `src/boosted_trees_course/xgb_style.py`

## Notebook checks and mastery questions

Use checks such as:

- histogram bin counts cover every selected row exactly once;
- gradient/Hessian bin sums match unbinned totals;
- finer binning does not reduce the set of available boundaries in the controlled setup; and
- the growth simulation always selects the advertised next leaf.

Ask the learner when leaf-wise growth is attractive, why binning can regularize as well as
accelerate, and which production improvements do not alter the loss objective.

## Deliverables

- [ ] Exact-versus-histogram split visual/experiment
- [ ] Depth-wise-versus-leaf-wise growth visual
- [ ] XGBoost/LightGBM design comparison table
- [ ] Explained aggregate checks
- [ ] Learner account of what the custom implementation omits and why

## Stopping condition

Stop when the learner can explain the production bridge without confusing objective,
split-search approximation, growth policy, and systems engineering. Do not run either
production library until Lesson 12.

## Optional extension

Inspect current official documentation for one version-sensitive default and record the
version/source, without turning the lesson into broad library research.
