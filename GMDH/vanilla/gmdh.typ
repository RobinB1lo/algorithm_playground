= Vanilla GMDH: Methodology and Implementation

== Overview

The Group Method of Data Handling (GMDH) is a self-organizing, inductive
algorithm for constructing regression models of complex, nonlinear systems
from limited data. It was developed by A. G. Ivakhnenko, first described in
"Heuristic Self-Organization in Problems of Engineering Cybernetics"
(Ivakhnenko, 1970), and elaborated with worked numerical examples in
"Polynomial Theory of Complex Systems" (Ivakhnenko, 1971). GMDH is designed
specifically to address two constraints that make ordinary regression
impractical: a large number of candidate input variables, and comparatively
few observations relative to the degree of polynomial needed to describe the
system accurately.

== Theoretical Basis: The Kolmogorov-Gabor Polynomial

The target that GMDH ultimately approximates is the *Kolmogorov-Gabor
polynomial*. The complete, arbitrarily high-degree multinomial expansion of
an output variable $y$ in terms of $n$ input variables $x_1, ..., x_n$:

$ y = a_0 + sum_(i=1)^n a_i x_i + sum_(i=1)^n sum_(j=i)^n a_(i j) x_i x_j
  + sum_(i=1)^n sum_(j=i)^n sum_(k=j)^n a_(i j k) x_i x_j x_k + dots $

For any nontrivial number of inputs, fitting this polynomial directly is
infeasible: the term count grows combinatorially (a ten-input system already
yields on the order of 200,000 terms), and estimating that many coefficients
would require far more data points than are typically available. Ivakhnenko
refers to this as the source of Bellman's "curse of multidimensionality."

GMDH's central idea is to *replace* the single, intractable complete
polynomial with a multilayered network of small, low-order *partial
polynomials*, each fit on only a pair of inputs at a time. Layering these
partial descriptions and selecting only the most accurate ones at each stage
reconstructs the effect of a much higher-degree polynomial without ever
needing to solve for it directly.

== Algorithm Description

The algorithm proceeds in five stages.

=== Step 1 — Data Partitioning

The dataset is split into two disjoint subsets:

- a *training* set, used to fit each candidate partial polynomial's
  coefficients, and
- a *selection* set (Ivakhnenko's "checking sequence"), used to score each
  fitted candidate on data it did not see during fitting.

This internal selection set is distinct from any external test set used
later to report final model accuracy; the latter is never seen by the model
during fitting or layer selection, whereas the selection set is used
repeatedly, once per layer, to decide which neurons survive.

=== Step 2 — Pairwise Partial Polynomials

At each layer, every pair of available input columns $(x_i, x_j)$ is
combined into a second-degree partial polynomial (a single "neuron"):

$ y = a_0 + a_1 x_i + a_2 x_j + a_3 x_i^2 + a_4 x_j^2 + a_5 x_i x_j $

This form — a truncated, two-variable instance of the Kolmogorov-Gabor
polynomial — is the "elementary algorithm" referred to in the original
paper. Coefficients $a_0, ..., a_5$ are estimated by least-squares
regression on the training set.

=== Step 3 — Threshold Self-Selection

Every candidate neuron from Step 2 is scored on the selection set using root
mean squared error (RMSE). Candidates are ranked, and only the
best-performing ones are retained to become inputs to the next layer; the
rest are discarded.

*On the survivor count.* In this implementation, the number of survivors is
a user-supplied constant, `n_keep`, fixed for every layer. This is a
deliberate simplification of Ivakhnenko's original scheme, in which the
survivor count is not fixed but is instead a *fraction* of that layer's
total candidates — approximately 40% at the first layer, decreasing sharply
in subsequent layers — and that fraction is itself found by searching
(Ivakhnenko uses Fibonacci search) for the value that minimizes error on the
selection set. Fixing `n_keep` as an absolute count, rather than reproducing
this fractional, searched threshold, is a documented departure from the
source algorithm, adopted here for implementation simplicity.

=== Step 4 — Layer Construction

The outputs of the surviving neurons (their predicted values, not their
coefficients) become the input columns for the next layer. New pairwise
combinations are formed from *these* outputs, and Step 3 is repeated. Each
successive layer therefore operates on progressively more processed,
higher-order information, without ever requiring the full Kolmogorov-Gabor
expansion to be constructed explicitly.

=== Step 5 — Termination and Model Selection

The layer-building process (Steps 2–4) repeats until one of the following
holds:

- a fixed ceiling on the number of layers (`max_layers`) is reached,
- the best selection-set error fails to improve for more layers than a
  configured tolerance (`patience`) — Ivakhnenko's "degeneracy," the point
  past which additional layers no longer improve, and typically worsen,
  held-out accuracy, or
- (optionally, if `method="threshold"`) a target error level is reached.

Layers built after the best-performing one are discarded. The single neuron
with the lowest selection-set error, across all retained layers, is used as
the final output node; predictions are made by replaying only the chain of
neurons that feeds into that node.

*On these stopping parameters.* None of `max_layers`, `patience`, `method`,
or `threshold` correspond to fixed values specified in the original paper.
Ivakhnenko does not set an a priori layer limit; he detects degeneracy
empirically from the shape of the error curve across layers. `max_layers`
here serves as a practical safety bound approximating that behavior, not a
reproduction of a literature-derived constant.

== Implementation Architecture

The implementation consists of three classes, corresponding respectively to
a single partial polynomial, a layer of such polynomials, and the network as
a whole.

=== `GMDHNeuron`

Represents one partial (second-degree, two-input) polynomial.

- `__init__(i, j)`: stores the column indices of the two inputs this neuron
  combines. No fitting occurs at construction time.
- `_features(a, b)`: constructs the design matrix
  $[1, a, b, a b, a^2, b^2]$ for the polynomial above.
- `fit(a, b, y, ridge)`: solves the (ridge-regularized) normal equations for
  the six coefficients, using only the *training* partition. The ridge term
  is a numerical-stability addition with no equivalent in the original
  paper; Ivakhnenko solves the un-regularized normal equations directly,
  relying on having sufficiently many interpolation points to keep the
  system well-conditioned.
- `predict(a, b)`: evaluates the fitted polynomial on new input values.
- `error(a, b, y)`: computes RMSE between predictions and true values —
  used exclusively on the *selection* partition, never on training data,
  to avoid evaluating a neuron on the same data used to fit it.

=== `GMDHLayer`

Manages one layer's worth of `GMDHNeuron` instances.

- `__init__(n_keep, ridge)`: stores the survivor count and ridge parameter
  for this layer; initializes an empty neuron list and a running best-error
  value.
- `fit(Z_tr, Z_se, y_tr, y_se)`: forms every pairwise combination of the
  layer's input columns, fits and scores each as a candidate neuron, retains
  the `n_keep` lowest-error survivors, and returns their outputs — computed
  on *both* the training and selection partitions — to serve as the next
  layer's inputs. Critically, `Z_tr` and `Z_se` are the *same* train/
  selection row-partition established once, at the top level, in `GMDH.fit`;
  no new splitting occurs here or within any individual neuron. Each neuron
  differs only in which two columns it reads, not in which rows it sees.
- `predict(Z)`: used only at inference time, after fitting is complete. It
  replays this layer's already-selected, already-fitted neurons on a new
  input matrix (e.g., an external test set never seen during fitting),
  producing this layer's output columns for the next layer's `predict` call.
  It performs no scoring or selection — that work was already done in
  `fit`; it exists purely to reproduce a layer's transformation on unseen
  data.

=== `GMDH`

Orchestrates the full multilayer network.

- `__init__(...)`: stores all hyperparameters (see Step 5 discussion above
  regarding which of these, if any, trace back to the source literature).
- `_standardize(X, y)`: standardizes inputs and target to zero mean, unit
  variance, storing the statistics needed to de-standardize predictions
  later. This is a modern numerical-conditioning step, not discussed in the
  original paper, which instead works with normalized deviations from
  fitted trend curves (a related but distinct form of preprocessing specific
  to Ivakhnenko's time-series examples).
- `fit(X, y)`: performs the single train/selection split described in Step
  1, then repeatedly builds layers (Step 2–4) and applies the stopping
  logic (Step 5), retaining only the layers up to and including the
  best-performing one.
- `predict(X)`: standardizes new input data using the statistics stored
  during `fit`, replays each retained layer's `predict` method in sequence,
  and de-standardizes the final output-node prediction back to the original
  scale of $y$.

== Summary of Departures from Ivakhnenko (1970, 1971)

For transparency, the following aspects of this implementation are
simplifications or additions relative to the source papers, rather than
direct reproductions:

- Fixed-count survivor selection (`n_keep`) in place of a searched,
  layer-decaying survival *fraction*.
- No internal search (e.g., Fibonacci search) over selection thresholds;
  thresholds are supplied, not optimized, by the fitting procedure itself.
- Ridge-regularized least squares in place of unregularized normal
  equations.
- Standardization of inputs/target in place of trend-deviation
  normalization.
- An explicit `max_layers` ceiling in place of purely empirical degeneracy
  detection.

== References

- Ivakhnenko, A. G. (1970). Heuristic Self-Organization in Problems of
  Engineering Cybernetics. #emph[Automatica], 6(2), 207–219.
- Ivakhnenko, A. G. (1971). Polynomial Theory of Complex Systems. #emph[IEEE
  Transactions on Systems, Man, and Cybernetics], SMC-1(4), 364–378.