# GitHub Issue Preview

> **Status:** Draft preview. No remote issues have been created yet.

## Labels

| Label | Color | Description |
|---|---|---|
| `lesson` | `#1D76DB` | A follow-along course lesson |
| `action: learner` | `#C2E0C6` | Requires learner explanation, implementation, or experiment |
| `phase: cart` | `#0E8A16` | Decision-tree foundations and CART implementation |
| `phase: boosting` | `#FBCA04` | First-order boosting and generalized losses |
| `phase: second-order` | `#D93F0B` | Second-order objective and compact XGBoost-style learner |
| `phase: libraries` | `#5319E7` | Production bridge and final library experiments |

## Lesson issues

# 01 — Lesson 01: See a regression tree as piecewise-constant predictions

**Labels:** `lesson`, `action: learner`, `phase: cart`

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

---

# 02 — Lesson 02: Derive and visualize CART split selection

**Labels:** `lesson`, `action: learner`, `phase: cart`

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

---

# 03 — Lesson 03: Grow trees recursively and control overfitting

**Labels:** `lesson`, `action: learner`, `phase: cart`

## Lesson metadata

- **Notebook:** [`notebooks/03-recursive-growth-and-regularization.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/03-recursive-growth-and-regularization.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** Best-split selection at one node
- **Depends on:** Lessons 01–02

## Primary objective

Explain how repeated local split selection produces a recursive tree and how stopping rules
control complexity, unstable leaves, and overfitting.

## Tutor must explain first

Before asking for structures or pseudocode, explain:

- the recursive base case and recursive split case;
- the difference between node depth, tree depth, samples at a node, and samples in a leaf;
- why greedy local choices do not globally optimize every possible tree;
- how `max_depth`, minimum samples per leaf, and minimum gain prevent a split; and
- why deeper trees lower training error but can worsen held-out behavior.

Present pre-pruning as the course's main regularization mechanism. Mention post-pruning only
as context, not a required implementation.

## Real-data intuition and required visual

Use a small California Housing slice with two features. Manually follow one root split and
one child split, keeping row membership visible at every node.

Draw a simple Matplotlib tree diagram beside the corresponding rectangular feature-space
regions. Add a compact depth/leaf-complexity comparison using a larger real-data subset; the
tutor may provide temporary scaffolding for this comparison without giving away Lesson 04's
estimator implementation.

## Derivation target

Write the recursive training rule in concise mathematical/pseudocode form and define the
conditions under which a node becomes a leaf. No formal convergence proof, cost-complexity
pruning derivation, or exhaustive global tree search.

## Learner-owned work

After the recursion is explained, have the learner:

- trace row indices through a hand-grown depth-two tree;
- define typed node representations (for example, leaf and split dataclasses) in `tree.py`;
- write clear recursive-growth pseudocode tied to those types; and
- reason through which stopping rule fires in several concrete node states.

Do not require the full `fit` implementation yet.

## Files touched

- `notebooks/03-recursive-growth-and-regularization.ipynb`
- `src/boosted_trees_course/tree.py`
- `src/boosted_trees_course/visualization.py` if a reusable tree diagram helper emerges

## Notebook checks and mastery questions

Use notebook assertions or explicit traces to check:

- left and right child memberships are disjoint and cover the parent;
- a stopped node has no children;
- minimum leaf size is respected in proposed splits; and
- depth accounting is consistent from the root.

Ask the learner why unlimited depth can isolate observations and why a tiny positive gain may
not justify another split.

## Deliverables

- [ ] Depth-two real-data tree/region walkthrough
- [ ] Concise recursive-growth and stopping explanation
- [ ] Typed node representations
- [ ] Stopping-rule examples and explained checks
- [ ] Learner explanation of the bias/variance tradeoff

## Stopping condition

Stop when the learner can trace recursion, identify every base case, and explain what each
regularization parameter prevents. Leave the complete estimator for Lesson 04.

## Optional extension

Discuss post-pruning conceptually by asking which sibling leaves might be collapsed; do not
implement a pruning algorithm.

---

# 04 — Lesson 04: Implement a typed CART regressor from first principles

**Labels:** `lesson`, `action: learner`, `phase: cart`

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

---

# 05 — Lesson 05: Diagnose and validate the custom CART regressor

**Labels:** `lesson`, `action: learner`, `phase: cart`

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

---

# 06 — Lesson 06: Understand boosting as sequential additive correction

**Labels:** `lesson`, `action: learner`, `phase: boosting`

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

---

# 07 — Lesson 07: Implement first-order gradient boosting from first principles

**Labels:** `lesson`, `action: learner`, `phase: boosting`

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

---

# 08 — Lesson 08: Generalize residuals to loss gradients and binary logits

**Labels:** `lesson`, `action: learner`, `phase: boosting`

## Lesson metadata

- **Notebook:** [`notebooks/08-general-losses-and-binary-classification.ipynb`](https://github.com/majorgilles/gradient-boosted-trees-course/blob/main/notebooks/08-general-losses-and-binary-classification.ipynb)
- **Estimated time:** 1–2 hours
- **Prerequisites:** First-order squared-error booster and derivative basics
- **Depends on:** Lessons 01–07

## Primary objective

Generalize “fit residuals” into “fit negative loss gradients” and use a focused binary
logistic-loss extension to show why gradient boosting is not fundamentally tied to ordinary
residuals.

## Tutor must explain first

Before asking for a loss abstraction, explain:

- loss as a function of the target and current raw prediction;
- derivative with respect to the current prediction, not model parameters;
- negative gradient as the desired first-order correction;
- logits versus probabilities and why boosting updates raw scores; and
- how sigmoid and binary log loss connect target labels to gradient direction.

Create the high-price target explicitly as a pedagogical label. When using a held-out split,
compute its threshold from training targets only.

## Real-data intuition and required visual

Use a tiny California Housing slice containing rows on both sides of the training-derived
high-price threshold.

Plot current probabilities and negative gradients for positive and negative rows at several
raw-score values. Include a small table showing how confidently wrong examples request
larger corrections than confidently correct examples.

## Derivation target

Derive only what is needed:

1. squared-error gradient and its residual equivalence;
2. sigmoid transformation from raw score to probability;
3. binary-log-loss gradient with respect to the raw score.

Map every symbol to a typed loss method and notebook array. Save the Hessian and Taylor
approximation for Lesson 09. Multiclass softmax is out of scope.

## Learner-owned work

After both losses are explained, have the learner:

- define a small typed first-order loss protocol/abstraction;
- implement squared-error and binary-log-loss values/gradients;
- refactor the first-order booster to request negative gradients from the loss; and
- run a compact binary boosting experiment using the derived real-data label.

Scaffold stable numerical helpers if needed, but explain clipping/stability before providing
them.

## Files touched

- `notebooks/08-general-losses-and-binary-classification.ipynb`
- `src/boosted_trees_course/losses.py`
- `src/boosted_trees_course/boosting.py`
- `src/boosted_trees_course/data.py` for leakage-safe label derivation

## Notebook checks and mastery questions

Use checks such as:

- probabilities stay within the expected range and preserve raw-score ordering;
- squared-error negative gradients match residuals on visible values;
- binary gradients point in the expected direction for positive/negative labels;
- the threshold is computed from training targets only; and
- selected first-order binary-loss values improve across early stages.

Ask the learner why fitting `y - probability` resembles residual fitting but occurs in raw
score space.

## Deliverables

- [ ] Typed first-order loss abstraction
- [ ] Squared-error and binary-log-loss implementations
- [ ] Leakage-safe high-price target helper
- [ ] Gradient-direction visual and explained assertions
- [ ] Focused binary boosting experiment and learner explanation

## Stopping condition

Stop when the learner can explain residuals as one special negative gradient and can trace a
binary label through raw score, probability, gradient, correction tree, and updated score.
Do not introduce Hessian-weighted gain yet.

## Optional extension

Numerically compare an analytic gradient with a small finite-difference estimate for a few
visible values; keep this a notebook sanity check, not a testing framework.

---

# 09 — Lesson 09: Derive second-order leaf weights and split gain

**Labels:** `lesson`, `action: learner`, `phase: second-order`

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

---

# 10 — Lesson 10: Implement a compact XGBoost-style learner

**Labels:** `lesson`, `action: learner`, `phase: second-order`

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

---

# 11 — Lesson 11: Connect the custom learner to XGBoost and LightGBM internals

**Labels:** `lesson`, `action: learner`, `phase: libraries`

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

---

# 12 — Lesson 12: Run final XGBoost and LightGBM experiments on California Housing

**Labels:** `lesson`, `action: learner`, `phase: libraries`

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

---
