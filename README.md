# Gradient-Boosted Trees from First Principles

A 12-lesson, tutor-guided mini-course for understanding decision trees, gradient boosting,
and the core ideas behind XGBoost and LightGBM.

The course is intentionally built as a follow-along project. Every notebook starts with only
a title. Open the corresponding GitHub issue and work with an AI tutor to build the
explanation, LaTeX derivations, illustrations, implementation, and notebook checks together.

## What you will learn

By the end of the course, you should be able to:

- explain how CART regression trees choose thresholds and leaf predictions;
- recognize how depth, minimum leaf size, and minimum gain control overfitting;
- implement a typed CART-style regressor from first principles;
- explain boosting as stage-wise additive optimization over predictions;
- implement a typed first-order gradient-boosting regressor;
- distinguish residuals from negative gradients for general losses;
- derive and implement a compact second-order XGBoost-style learner;
- explain gradients, Hessians, regularized leaf weights, and split gain;
- connect the educational implementation to XGBoost and LightGBM design choices; and
- run controlled XGBoost and LightGBM experiments on California Housing.

The custom models prioritize clarity and inspectability. They are not production-library
replicas.

## Course workflow

For each lesson:

1. Open its GitHub issue and read the objective, prerequisites, and stopping condition.
2. Ask the AI tutor to read [`AGENTS.md`](AGENTS.md), the issue, and the active notebook.
3. Co-create only that lesson's notebook; do not prefill future notebooks.
4. Move reusable, typed logic into `src/boosted_trees_course/` as directed by the issue.
5. Use explained notebook assertions, prediction questions, and plots to check behavior.
6. Record your own explanation and takeaways before closing the issue.

## Lessons

| # | Lesson | Notebook | Issue |
|---:|---|---|---|
| 01 | Tree prediction geometry and terminology | [`01-tree-prediction-geometry.ipynb`](notebooks/01-tree-prediction-geometry.ipynb) | [#1](https://github.com/majorgilles/gradient-boosted-trees-course/issues/1) |
| 02 | CART split objective | [`02-cart-split-objective.ipynb`](notebooks/02-cart-split-objective.ipynb) | [#2](https://github.com/majorgilles/gradient-boosted-trees-course/issues/2) |
| 03 | Recursive growth and regularization | [`03-recursive-growth-and-regularization.ipynb`](notebooks/03-recursive-growth-and-regularization.ipynb) | [#3](https://github.com/majorgilles/gradient-boosted-trees-course/issues/3) |
| 04 | Implement a CART regressor | [`04-implement-cart-regressor.ipynb`](notebooks/04-implement-cart-regressor.ipynb) | [#4](https://github.com/majorgilles/gradient-boosted-trees-course/issues/4) |
| 05 | Diagnose CART behavior | [`05-diagnose-cart-behavior.ipynb`](notebooks/05-diagnose-cart-behavior.ipynb) | [#5](https://github.com/majorgilles/gradient-boosted-trees-course/issues/5) |
| 06 | Boosting as additive correction | [`06-boosting-as-additive-correction.ipynb`](notebooks/06-boosting-as-additive-correction.ipynb) | [#6](https://github.com/majorgilles/gradient-boosted-trees-course/issues/6) |
| 07 | Implement first-order gradient boosting | [`07-implement-first-order-gradient-boosting.ipynb`](notebooks/07-implement-first-order-gradient-boosting.ipynb) | [#7](https://github.com/majorgilles/gradient-boosted-trees-course/issues/7) |
| 08 | General losses and binary classification | [`08-general-losses-and-binary-classification.ipynb`](notebooks/08-general-losses-and-binary-classification.ipynb) | [#8](https://github.com/majorgilles/gradient-boosted-trees-course/issues/8) |
| 09 | Second-order objective and split gain | [`09-second-order-objective-and-split-gain.ipynb`](notebooks/09-second-order-objective-and-split-gain.ipynb) | [#9](https://github.com/majorgilles/gradient-boosted-trees-course/issues/9) |
| 10 | Implement an XGBoost-style learner | [`10-implement-xgboost-style-learner.ipynb`](notebooks/10-implement-xgboost-style-learner.ipynb) | [#10](https://github.com/majorgilles/gradient-boosted-trees-course/issues/10) |
| 11 | XGBoost and LightGBM production bridge | [`11-xgboost-lightgbm-production-bridge.ipynb`](notebooks/11-xgboost-lightgbm-production-bridge.ipynb) | [#11](https://github.com/majorgilles/gradient-boosted-trees-course/issues/11) |
| 12 | Final XGBoost and LightGBM experiments | [`12-final-xgboost-lightgbm-experiments.ipynb`](notebooks/12-final-xgboost-lightgbm-experiments.ipynb) | [#12](https://github.com/majorgilles/gradient-boosted-trees-course/issues/12) |

## Setup

Requirements:

- [`uv`](https://docs.astral.sh/uv/)
- a first-use network connection for Python/packages and California Housing

```bash
uv python install 3.14
uv sync
uv run jupyter lab
```

The project requires Python `>=3.14,<3.15`. The default environment includes NumPy,
pandas, Matplotlib, scikit-learn, JupyterLab, ipykernel, and Ruff.

XGBoost and LightGBM are isolated in the final-course dependency group. Install them only
when you reach Lesson 12:

```bash
uv sync --group final
```

## California Housing data

The course uses `sklearn.datasets.fetch_california_housing` as one continuous real-data
story. Small deterministic row-and-feature slices make calculations inspectable; larger
subsets and the full data support later experiments.

scikit-learn downloads the dataset on first use and caches it outside this repository
(normally under `~/scikit_learn_data`). The raw dataset is not committed. If you are offline
before the first successful fetch, follow the active lesson issue with the tutor after a
network connection is available.

A focused classification lesson derives an explicitly pedagogical high-price label. Its
threshold must be learned from training data, not from held-out data.

## Lightweight project checks

The course primarily uses transparent notebook assertions, hand calculations, fixed seeds,
loss curves, and visual diagnosis. Focused pytest checks protect reusable helpers moved into
`src/boosted_trees_course/` without turning the course into a production test suite.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Repository layout

```text
.
├── AGENTS.md                     # tutor and coding contract
├── README.md
├── docs/
│   └── shared-understanding.md   # course decisions and boundaries
├── notebooks/                    # 12 title-only lesson starters
├── src/boosted_trees_course/     # cumulative typed implementation
├── artifacts/figures/            # durable figures only
└── pyproject.toml                # uv groups and Ruff configuration
```

## Course boundaries

- No production-scale clone of XGBoost or LightGBM.
- No multiclass implementation requirement.
- No GPU, distributed training, or performance-engineering detour.
- No synthetic-data requirement; use curated California Housing slices.
- No proof-heavy mathematics that does not improve implementation intuition.
- No full lessons or reference solutions pre-authored in starter notebooks.
