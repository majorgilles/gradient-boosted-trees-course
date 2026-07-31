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
