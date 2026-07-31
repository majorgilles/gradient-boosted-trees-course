# Shared Understanding

## Course purpose

Build a personal, tutor-guided mini-course that starts from decision-tree internals, derives
and implements gradient boosting from first principles, adds a compact second-order
XGBoost-style learner, and only then uses XGBoost and LightGBM in controlled final
experiments.

## Learner and learning mode

- The course is primarily for one learner following along with an AI tutor.
- Assumed background: fluent Python, practical machine learning, gradient descent from deep
  learning, and experience using decision trees.
- Tree and boosted-tree internals are taught carefully rather than assumed.
- The tutor behaves as a collaborative pair programmer and teaches before assigning work.

## Completion evidence

The learner should be able to:

1. explain CART split selection, leaf prediction, recursion, overfitting, and regularization;
2. implement a typed CART regression tree;
3. explain boosting as stage-wise additive optimization over predictions;
4. implement a typed first-order gradient-boosting regressor;
5. distinguish residuals from negative gradients under general losses;
6. derive and implement gradients, Hessians, regularized leaf weights, and split gain in a
   compact XGBoost-style learner;
7. connect the educational implementation to XGBoost and LightGBM growth/split strategies;
8. compare the custom models with XGBoost and LightGBM on California Housing.

## Algorithm scope

### Implement

- CART-style regression splits, nodes, recursion, stopping, and prediction
- first-order gradient boosting
- a focused binary logistic-loss extension
- a compact second-order XGBoost-style objective and learner
- educational leaf regularization and split-gain calculations

### Explain and visualize, but do not fully recreate

- exact versus histogram/approximate splits
- depth-wise versus leaf-wise growth
- missing-value and sparse-feature routing
- row/feature sampling
- parallel, GPU, and distributed engineering

Multiclass boosting and production estimator completeness are out of scope.

## Mathematical and visual style

Each important mechanism should move through three stages:

1. a tiny real-data slice and geometric or visual intuition;
2. one concise, step-by-step LaTeX derivation in the notebook;
3. a direct mapping from symbols to typed Python variables and behavior.

Use only the calculus refreshers needed for gradients, Hessians, and Taylor approximation.
Avoid proof-heavy formalism that does not strengthen intuition.

Use static, reproducible Matplotlib visuals. Save only durable figures under
`artifacts/figures/`; use animation only when sequential correction is materially clearer.

## Dataset strategy

Use California Housing for the complete course story.

- Tiny deterministic row/feature slices support hand calculations.
- Larger subsets and full data support overfitting and final experiments.
- A clearly labeled high-price binary target supports the focused classification extension.
- Compute that threshold from training data only during held-out evaluation.
- Document scikit-learn's first-use download/cache behavior; do not commit the large raw
  dataset.

## Lesson and tutoring contract

- Approximately 12 issues, each targeting 1–2 hours.
- One primary objective and one clear stopping point per issue.
- Every issue is a detailed tutor brief, not a finished lesson or reference solution.
- Every starter notebook contains only a title.
- The learner and tutor co-create explanations, LaTeX, visuals, exercises, code, assertions,
  and reflection while following the issue.
- Partial code and fill-in gaps are used only when useful and only after the task has been
  explained.
- Reusable typed code evolves cumulatively under `src/boosted_trees_course/`.

## Validation policy

There is no pytest suite and no heavy automated-testing curriculum. Use:

- hand-computable examples;
- small explained notebook assertions;
- prediction-before-running questions;
- simple invariants and finite/shape checks;
- staged-loss and regularization behavior;
- plots and fixed-seed final comparisons.

“Tests with XGBoost and LightGBM” means controlled experiments and metric/behavior
comparisons, not exact parity tests.

## Technical decisions

- Public GitHub repository: `majorgilles/gradient-boosted-trees-course`
- Local path: `C:/Users/giloz/dev/gradient-boosted-trees-course`
- Python: `>=3.14,<3.15`
- Package/environment management: uv
- Core dependencies: NumPy, pandas, Matplotlib, scikit-learn
- Development group: JupyterLab, ipykernel, Ruff
- Final group: XGBoost, LightGBM
- Ruff enforces annotations in `src`; notebooks relax annotation rules.
- No pytest and no separate static type checker.

Python 3.14 was selected because Python 3.15 remains prerelease until October 1, 2026 and
the required scientific/boosting packages currently declare support through Python 3.14.
See [PEP 790](https://peps.python.org/pep-0790/),
[NumPy](https://pypi.org/project/numpy/),
[scikit-learn](https://pypi.org/project/scikit-learn/),
[XGBoost](https://pypi.org/project/xgboost/), and
[LightGBM](https://pypi.org/project/lightgbm/).

## Key risks and mitigations

- **Scope pressure:** keep one major objective per issue.
- **Title-only notebooks:** make issues and `AGENTS.md` sufficiently detailed.
- **Cumulative code:** state dependencies and expected package state in every issue.
- **No synthetic data:** curate real rows/features for transparent arithmetic.
- **Notebook-only checks:** favor tiny inspectable cases and reproducible execution.
- **Dataset download:** document caching and fail clearly if offline before first fetch.
- **Derived classification label:** present it as a pedagogical device, not a policy claim.
- **New Python baseline:** constrain Python and commit `uv.lock`.
