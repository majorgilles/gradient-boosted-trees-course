## Lesson metadata

- **Notebook:** [`notebooks/05-diagnose-cart-behavior.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/05-diagnose-cart-behavior.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** A working custom CART regressor
- **Depends on:** Lessons 01–04

## Primary objective

Validate and diagnose the custom CART regressor through transparent notebook checks and
visual experiments that connect regularization settings to model behavior.

## Tutor must explain first

Before running experiments, explain:

- the difference between an implementation sanity check and a model-quality comparison;
- why training and validation error answer different questions;
- how depth, minimum leaf size, and minimum gain change partition granularity;
- which edge cases should become a single leaf; and
- why exact parity with another tree implementation is not a meaningful requirement.

## Real-data intuition and required visual

Create a fixed train/validation split from California Housing. Use a manageable subset for the
custom exhaustive implementation.

Produce at least two linked visuals:

1. training and validation error against tree depth or leaf count;
2. shallow-tree predictions/partitions that show how a regularization change alters regions.

Use the same data split across settings.

## Derivation target

Define the chosen regression metrics concisely and relate squared training objective to the
reported validation metric. No statistical learning theory proof, confidence intervals, or
hyperparameter optimization framework.

## Learner-owned work

Have the learner design and run a small controlled experiment over one complexity parameter
at a time. Add notebook checks for important edge cases and inspect failed behavior by tracing
nodes or predictions.

A comparison with scikit-learn's `DecisionTreeRegressor` may be used only as a labeled
behavioral reference after the custom model is checked. Do not demand equal trees or exact
predictions.

## Files touched

- `notebooks/05-diagnose-cart-behavior.ipynb`
- `src/boosted_trees_course/tree.py` for fixes discovered through diagnosis
- `src/boosted_trees_course/visualization.py` for reusable diagnostics

## Notebook checks and mastery questions

Include explained assertions for selected cases such as:

- constant target → one effective prediction value;
- no valid feature variation → leaf rather than crash;
- finite predictions with the expected shape;
- configured depth and minimum-leaf constraints are respected; and
- train/validation splitting is reproducible.

Ask the learner to predict the train/validation curves before plotting and explain an example
where more leaves hurt held-out performance.

## Deliverables

- [ ] Fixed-seed real-data validation experiment
- [ ] Complexity-versus-error visual
- [ ] Edge-case notebook assertions
- [ ] Any diagnosed fixes in the typed CART module
- [ ] Learner explanation of underfitting and overfitting in tree terms

## Stopping condition

Stop when the learner trusts the custom CART implementation on transparent cases and can use
visual/metric evidence to choose a reasonable tree complexity. Do not start boosting.

## Optional extension

Inspect prediction paths for two nearby houses that land in different leaves and explain the
threshold responsible.
