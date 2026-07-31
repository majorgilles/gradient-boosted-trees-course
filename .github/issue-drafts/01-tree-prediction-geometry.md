## Lesson metadata

- **Notebook:** [`notebooks/01-tree-prediction-geometry.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/01-tree-prediction-geometry.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Python, NumPy/pandas basics, and experience calling a decision-tree model
- **Depends on:** Nothing

## Primary objective

Explain a regression tree as a piecewise-constant prediction function and use the language of
features, thresholds, branches, internal nodes, leaves, regions, and leaf predictions
precisely.

## Tutor must explain first

Before asking for code, explain:

- what a regression feature and continuous target represent in California Housing;
- how an axis-aligned threshold partitions observations;
- why every observation reaching one leaf receives the same prediction;
- why a leaf mean is a sensible regression prediction, without yet optimizing a split; and
- how a tree diagram and a step-function plot describe the same model.

Diagnose the learner's existing tree knowledge instead of assuming knowledge of CART
internals.

## Real-data intuition and required visual

Load California Housing and select a deterministic, documented slice of roughly 10–16 rows
spanning the `MedInc` feature. Keep the target and one feature visible in a small table.

Create a two-panel Matplotlib figure:

1. target value against `MedInc`, with one manually chosen threshold and colored left/right
   regions;
2. the corresponding piecewise-constant predictions, with leaf means drawn as horizontal
   steps.

The slice must contain real rows, not generated observations.

## Derivation target

Introduce notation only for a threshold rule, its left/right regions, and the mean prediction
inside a region. Keep this as a symbol-to-picture mapping; do **not** derive split
optimization, impurity reduction, recursion, or boosting yet.

## Learner-owned work

After the concept is explained, have the learner:

- choose or reason about a transparent threshold;
- compute the two leaf means from the visible rows;
- implement a small one-feature stump prediction function in the notebook; and
- explain why moving the threshold can change both group membership and both predictions.

A partial function is appropriate only after its inputs, output shape, and rule are explicit.

## Files touched

- `notebooks/01-tree-prediction-geometry.ipynb`
- `src/boosted_trees_course/data.py` for a typed fetch/slice helper if reuse is already useful
- `src/boosted_trees_course/visualization.py` only for a clearly reusable plotting helper

## Notebook checks and mastery questions

Use small, explained checks such as:

- every visible row is assigned to exactly one side;
- the prediction vector has the expected shape;
- predictions contain at most the two computed leaf values.

Ask the learner to predict before running:

- what happens to a point when it crosses the threshold;
- whether two points in one leaf can receive different predictions; and
- how the tree diagram maps to the step-function plot.

## Deliverables

- [ ] Real California Housing slice with documented selection
- [ ] Threshold/region scatter plot and step-function prediction plot
- [ ] Small stump prediction implementation
- [ ] Explained notebook assertions
- [ ] Learner-authored definitions and takeaway

## Stopping condition

Stop when the learner can translate fluently among a threshold rule, a two-leaf tree diagram,
and a piecewise-constant prediction plot. Do not begin best-split search.

## Optional extension

Use a second feature only to show that axis-aligned trees create rectangular regions; do not
turn this into a split-search exercise.
