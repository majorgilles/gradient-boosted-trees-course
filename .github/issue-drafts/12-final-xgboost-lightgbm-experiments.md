## Lesson metadata

- **Notebook:** [`notebooks/12-final-xgboost-lightgbm-experiments.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/12-final-xgboost-lightgbm-experiments.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** All custom learners and the production-design bridge
- **Depends on:** Lessons 01–11

## Primary objective

Use XGBoost and LightGBM for the first time in a fair, fixed-seed California Housing
experiment and explain observed behavior through the concepts implemented earlier.

## Tutor must explain first

Before fitting models, explain:

- how to install the `final` uv dependency group;
- how equivalent-looking parameters can have different names/defaults/semantics;
- why all models must share one train/validation split and evaluation protocol;
- why the slow educational models may need a smaller training subset;
- what RMSE, MAE, training time, and validation curves do and do not establish; and
- why controlled comparisons are not exact implementation-parity tests.

Check the installed library versions and consult their current documentation before making
default-specific claims.

## Real-data experiment and required visuals

Use one fixed California Housing train/validation split. Fit a deliberately small, documented
set of models/configurations:

- a simple constant or shallow-tree baseline;
- the custom first-order or second-order learner on a manageable subset;
- XGBoost on the agreed training data; and
- LightGBM on the same agreed training data.

Keep the comparison focused rather than launching a tuning search. Produce:

1. a metrics/runtime table with honest dataset-size notes;
2. validation prediction or residual plots on the common held-out set;
3. learning/iteration curves where available and comparable; and
4. one controlled regularization or growth-policy experiment tied to earlier intuition.

## Derivation target

No new derivation. Map selected library parameters back to concepts already derived: learning
rate, number of trees, depth/leaves, minimum gain, leaf regularization, row/feature sampling,
and histogram/growth policy. Identify where mappings are only approximate.

## Learner-owned work

Have the learner:

- define the fair experiment before running it;
- build the fixed split and baseline;
- configure small XGBoost and LightGBM regressors intentionally rather than copying a large
  tuning grid;
- collect metrics and timing consistently;
- explain differences using objective, capacity, regularization, split search, and growth
  policy; and
- record at least one result that challenged an initial prediction.

Use library APIs now, but do not replace conceptual interpretation with parameter recipes.

## Files touched

- `notebooks/12-final-xgboost-lightgbm-experiments.ipynb`
- optionally reusable experiment plotting/data helpers in `src/boosted_trees_course/`
- `artifacts/figures/` for a small number of durable capstone figures

## Notebook checks and mastery questions

Use checks such as:

- train/validation row indices are disjoint and shared across comparable runs;
- features and targets are finite with expected shapes;
- all reported validation metrics use the same held-out targets;
- seeds, package versions, parameters, and data sizes are recorded; and
- predictions are finite and metric tables contain the intended models.

Ask the learner to explain:

- why the educational model is slower;
- why LightGBM and XGBoost may grow different trees under similar headline settings;
- which regularization parameter affects leaf value versus split acceptance;
- when a smaller learning rate needs more trees; and
- what evidence would be needed before preferring one library in a real project.

## Deliverables

- [ ] Reproducible fixed-seed experiment definition
- [ ] Baseline, custom-model, XGBoost, and LightGBM results
- [ ] Metrics/runtime table and required visuals
- [ ] Parameter-to-concept mapping
- [ ] Learner-authored capstone explanation and remaining questions

## Stopping condition

The course is complete when the learner can explain the full path from CART regions to
first-order corrections, second-order gain, production split/growth choices, and the observed
XGBoost/LightGBM results—without relying on library API descriptions alone.

## Optional extension

Repeat a compact comparison on the derived high-price target only if the regression capstone
and explanation are already complete; do not let it become an additional tuning project.
