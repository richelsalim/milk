# DMD Methods Worth Exploring for a CPU-Only Recommender System

## Purpose

This note is a **self-contained handoff document** for a future model or
engineer who will **not have access to the DMD textbook**. It extracts
the Dynamic Mode Decomposition (DMD) ideas most relevant to a CPU-only
recommender-system experiment, especially a temporally split interaction
dataset such as KuaiRand.

The intended use is **not** to replace a recommender with DMD. The
strongest initial framing is:

> **Use DMD-family methods to extract temporal/dynamical features, then
> feed those features into a conventional ranking model such as a
> Factorization Machine (FM).**

The recommended exploration order is:

1.  Classical low-rank DMD
2.  Delay-coordinate / Hankel DMD
3.  Multiresolution DMD (mrDMD)
4.  Sparsity-promoting DMD
5.  Noise-aware DMD variants
6.  DMD with control (DMDc), only if logged exposures/actions can be
    treated as inputs
7.  Extended/kernel DMD, only after linear DMD shows useful signal

------------------------------------------------------------------------

# 1. Why DMD may fit recommendation

A conventional recommender often models a relatively static relation

\[ (u,i,x) `\mapsto `{=tex}s\_{ui}, \]

where (u) is a user, (i) is an item, (x) contains side features, and
(s\_{ui}) is a ranking score.

DMD instead asks whether a measured state evolves approximately as

\[ x\_{k+1} `\approx `{=tex}A x_k. \]

Here, (x_k) is a **snapshot of the system at time (k)**. For
recommendation, a snapshot should not naively be the entire dense
user-item matrix. It should be a CPU-manageable representation such as:

-   per-category interaction rates,
-   per-category long-view rates,
-   item/popularity aggregates,
-   user-cluster activity,
-   latent factors or compressed interaction features,
-   multi-feedback aggregates such as click/like/follow/long-view rates.

DMD is attractive because it combines low-rank dimensionality reduction
with temporal evolution. Its principal computational operation is an
SVD, making low-rank versions practical on CPU.

------------------------------------------------------------------------

# 2. Classical low-rank DMD --- implement first

## 2.1 Snapshot construction

Given snapshots

\[ x_1,x_2,`\ldots`{=tex},x_m, \]

construct the time-shifted matrices

\[ X =
```{=tex}
\begin{bmatrix}
x_1 & x_2 & \cdots & x_{m-1}
\end{bmatrix}
```
, \]

\[ X' =
```{=tex}
\begin{bmatrix}
x_2 & x_3 & \cdots & x_m
\end{bmatrix}
```
. \]

DMD fits

\[ X' `\approx `{=tex}AX. \]

The least-squares solution is

\[ A = X'X\^`\dagger`{=tex}, \]

where (X\^`\dagger`{=tex}) is the Moore-Penrose pseudoinverse.

Do **not** normally form (A) explicitly when the state dimension is
large.

## 2.2 Low-rank algorithm

Compute a rank-(r) truncated SVD:

\[ X `\approx `{=tex}U_r`\Sigma`{=tex}\_rV_r\^\*. \]

Construct the reduced evolution operator

\[ `\tilde `{=tex}A = U_r\^\*X'V_r`\Sigma`{=tex}\_r\^{-1}. \]

Eigendecompose it:

\[ `\tilde `{=tex}A W = W`\Lambda`{=tex}. \]

The exact DMD modes are

\[ `\Phi`{=tex} = X'V_r`\Sigma`{=tex}\_r\^{-1}W. \]

The diagonal entries of (`\Lambda`{=tex}) are the discrete-time DMD
eigenvalues (`\lambda`{=tex}\_j).

For time step (`\Delta `{=tex}t), continuous-time eigenvalues are

\[ `\omega`{=tex}\_j = `\frac{\log(\lambda_j)}{\Delta t}`{=tex}. \]

Given the initial state (x_1), estimate modal amplitudes

\[ b=`\Phi`{=tex}\^`\dagger `{=tex}x_1. \]

Then reconstruct or forecast

\[ x(t) `\approx`{=tex} `\Phi `{=tex}`\exp`{=tex}(`\Omega `{=tex}t)b, \]

where

\[
`\Omega`{=tex}=`\operatorname{diag}`{=tex}(`\omega`{=tex}\_1,`\ldots`{=tex},`\omega`{=tex}\_r).
\]

Equivalently in discrete time,

\[ x_k `\approx `{=tex}`\Phi`{=tex}`\Lambda`{=tex}\^{k-1}b. \]

## 2.3 Interpretation

For a mode (j):

-   (\|`\lambda`{=tex}\_j\|\>1): growing mode.
-   (\|`\lambda`{=tex}\_j\|\<1): decaying mode.
-   (\|`\lambda`{=tex}\_j\|`\approx1`{=tex}): persistent mode.
-   complex phase / imaginary part of (`\omega`{=tex}\_j): oscillatory
    or periodic behavior.

In recommendation language, modes may correspond to persistent tastes,
emerging trends, decaying interests, or periodic behavior.

## 2.4 Recommended first experiment

Do **not** make DMD itself output the final ranking.

Use:

``` text
interaction logs
      |
time-window aggregation
      |
X, X'
      |
low-rank DMD
      |
DMD temporal features
      |
FM / ranking model
      |
GAUC + nDCG
```

Useful DMD-derived features may include:

-   forecasted activity for a category/item group,
-   mode amplitudes,
-   dominant growth/decay rates,
-   distance between current and DMD-predicted state,
-   projections onto selected DMD modes,
-   short-horizon trend estimates.

Always compare against a simpler temporal baseline such as raw recency,
rolling counts, and exponentially decayed popularity.

------------------------------------------------------------------------

# 3. Delay-coordinate / Hankel DMD --- highest-priority extension

Classical DMD uses a first-order state:

\[ x\_{k+1}`\approx `{=tex}Ax_k. \]

User behavior may depend on a longer history. Delay coordinates augment
the state with past observations.

For (s) delays, define

\[ z_k =
```{=tex}
\begin{bmatrix}
x_k\\
x_{k+1}\\
\vdots\\
x_{k+s-1}
\end{bmatrix}
```
. \]

Equivalently, construct shift-stacked matrices

\[ X\_{`\mathrm{aug}`{=tex}} =
```{=tex}
\begin{bmatrix}
x_1 & x_2 & \cdots & x_{m-s}\\
x_2 & x_3 & \cdots & x_{m-s+1}\\
\vdots & \vdots & \ddots & \vdots\\
x_s & x_{s+1} & \cdots & x_{m-1}
\end{bmatrix}
```
, \]

\[ X'\_{`\mathrm{aug}`{=tex}} =
```{=tex}
\begin{bmatrix}
x_2 & x_3 & \cdots & x_{m-s+1}\\
x_3 & x_4 & \cdots & x_{m-s+2}\\
\vdots & \vdots & \ddots & \vdots\\
x_{s+1} & x_{s+2} & \cdots & x_m
\end{bmatrix}
```
. \]

Then apply ordinary DMD to

\[
X'*{`\mathrm{aug}`{=tex}}`\approx `{=tex}A*{`\mathrm{aug}`{=tex}}X\_{`\mathrm{aug}`{=tex}}.
\]

## Why it is promising for recommendation

Delay embedding can encode:

-   recency,
-   momentum,
-   repeated interests,
-   short behavioral sequences,
-   periodicity,
-   transitions that cannot be inferred from one snapshot.

It can also increase the effective numerical rank when the original
measurement vector is too small to represent the dynamics.

## CPU considerations

If snapshot dimension is (n), delay depth (s) makes the augmented
dimension approximately (ns). Keep (n) and (s) modest.

Initial search:

-   (s`\in`{=tex}{2,3,5,7})
-   rank (r`\in`{=tex}{5,10,20,40})

Use truncated/randomized SVD if needed.

## Recommended hypothesis

> A short history of compressed user/item activity contains predictive
> temporal structure beyond static and one-step DMD features.

Compare:

1.  FM
2.  FM + simple recency
3.  FM + classical DMD
4.  FM + delay-DMD

------------------------------------------------------------------------

# 4. Hankel matrices and ERA connection

Delay-coordinate methods naturally produce Hankel-like matrices. A
generic Hankel matrix has time-shifted rows:

\[ H=
```{=tex}
\begin{bmatrix}
y_1 & y_2 & \cdots & y_q\\
y_2 & y_3 & \cdots & y_{q+1}\\
\vdots & \vdots & \ddots & \vdots\\
y_s & y_{s+1} & \cdots & y_{q+s-1}
\end{bmatrix}
```
. \]

Its SVD,

\[ H`\approx `{=tex}U_r`\Sigma`{=tex}\_rV_r\^\*, \]

extracts dominant temporal patterns. These modes can be interpreted as
low-dimensional temporal building blocks.

This is useful when the measured recommender state is only a partial
observation of a larger latent behavioral process.

Practical interpretation:

> Hankelization converts a short observed sequence into a
> higher-dimensional state so that simple linear dynamics can represent
> richer temporal behavior.

------------------------------------------------------------------------

# 5. Multiresolution DMD (mrDMD) --- second major extension

Classical DMD assumes one set of modes can describe the entire
observation window. That is weak when modes are transient or exist on
different time scales.

Recommendation naturally contains multiple scales:

``` text
long-term taste      ------------------------>
medium-term interest       ------------>
short-lived trend                  --->
```

mrDMD recursively partitions the time interval and performs DMD within
smaller windows.

Conceptually:

``` text
Level 1: |---------------- entire history ----------------|
Level 2: |-------------| |-------------|
Level 3: |------|------| |------|------|
...
```

At each level, slow modes are separated from faster/localized behavior.

## Why it matters

Potential recommendation signals include:

-   long-term stable preference,
-   weekly or periodic behavior,
-   medium-term topic interest,
-   short-lived viral trends,
-   abrupt interest changes.

Classical DMD can struggle with transient "turn-on / turn-off" dynamics.
mrDMD is specifically intended to localize dynamics in time and
frequency.

## Feature strategy

Rather than reconstructing the entire interaction system, derive
features such as:

\[ f\_{`\text{long}`{=tex}}, `\quad`{=tex} f\_{`\text{medium}`{=tex}},
`\quad`{=tex} f\_{`\text{short}`{=tex}}, \]

for each category/item group/user cluster.

Then feed them into the ranking model.

## Important warning

mrDMD adds hyperparameters:

-   number of levels (L),
-   DMD rank (r),
-   slow/fast eigenvalue threshold,
-   minimum samples per window.

Do not introduce it before classical/delay DMD demonstrates useful
temporal signal.

------------------------------------------------------------------------

# 6. Sparsity-promoting DMD --- third major extension

A DMD model may produce many modes, not all of which are useful.

Suppose

\[ X`\approx`{=tex}`\Phi `{=tex}D_b V\_{`\mathrm{and}`{=tex}}, \]

where (b) contains mode amplitudes. Instead of retaining every mode,
sparsity-promoting DMD chooses a small subset.

A conceptual objective is

\[ `\min`{=tex}\_b J(b)+`\gamma`{=tex}\|b\|\_0, \]

where (J(b)) is reconstruction error and (\|b\|\_0) counts nonzero
amplitudes.

Because this is combinatorial/nonconvex, replace it with the convex
relaxation

\[ `\boxed{
\min_b
J(b)+\gamma\|b\|_1
}`{=tex} \]

where (`\gamma`{=tex}) controls the accuracy-sparsity tradeoff.

Larger (`\gamma`{=tex}) generally encourages fewer active modes.

## Why it may help recommendation

Recommendation logs are noisy and sparse. Sparse mode selection can:

-   suppress weak/noisy temporal modes,
-   improve interpretability,
-   reduce feature count,
-   reduce downstream CPU cost,
-   regularize the DMD feature extractor.

## Important distinction

There are two different kinds of sparsity:

1.  **Sparse observations** --- few user-item interactions are observed.
2.  **Sparse DMD modes** --- only a few dynamical modes are retained.

Do not confuse them.

------------------------------------------------------------------------

# 7. Compressed DMD / compressed sensing --- lower priority but CPU-friendly

Compressed sensing assumes a signal has a sparse representation

\[ x=`\Psi `{=tex}s \]

with sparse (s), but only compressed measurements

\[ y=Cx=C`\Psi `{=tex}s \]

are observed.

A common reconstruction objective is

\[ `\hat `{=tex}s = `\arg`{=tex}`\min`{=tex}\_s
\|C`\Psi `{=tex}s-y\|\_2\^2 + `\lambda`{=tex}\|s\|\_1. \]

Compressed DMD applies related ideas so dynamics can be inferred from
limited measurements.

## Potential recommender use

Useful if the state representation becomes too large:

-   sample only selected categories/item groups,
-   use a compressed projection of interaction states,
-   run DMD in compressed coordinates.

However, if ordinary low-rank DMD already fits in RAM and CPU time,
compressed DMD is unnecessary complexity.

------------------------------------------------------------------------

# 8. Noise-aware DMD --- important if DMD looks unstable

Interaction logs contain substantial stochasticity. Classical DMD is a
least-squares fit and can be sensitive to noise.

The DMD book emphasizes:

-   singular-value truncation,
-   singular-value thresholding,
-   compensation for noise in the DMD spectrum,
-   total-least-squares / forward-backward style corrections.

## Practical first defense: rank truncation

From

\[ X=U`\Sigma `{=tex}V\^\*, \]

inspect singular values

\[
`\sigma`{=tex}\_1`\ge`{=tex}`\sigma`{=tex}\_2`\ge`{=tex}`\cdots`{=tex}.
\]

Choose (r) before the spectrum becomes dominated by noise.

Avoid setting (r) unnecessarily high.

## Total least-squares idea

Ordinary least squares effectively treats error asymmetrically. Total
least squares allows noise in both sides of

\[ X'`\approx `{=tex}AX. \]

This may be more realistic because both current and next temporal
snapshots are noisy behavioral aggregates.

## When to explore

Only after observing one of these symptoms:

-   eigenvalues change drastically across seeds/windows,
-   forecasts explode,
-   mode rankings are unstable,
-   small data perturbations cause large metric changes.

------------------------------------------------------------------------

# 9. DMD with Control (DMDc) --- conceptually powerful, but later

DMDc extends

\[ x\_{k+1}=Ax_k \]

to

\[ `\boxed{
x_{k+1}=Ax_k+Bu_k
}`{=tex} \]

where (u_k) represents an external input/control.

Construct

\[ X' = AX+B`\Upsilon`{=tex}, \]

where

\[ `\Upsilon`{=tex}=
```{=tex}
\begin{bmatrix}
u_1&u_2&\cdots&u_{m-1}
\end{bmatrix}
```
. \]

Stack state and input:

\[ `\Omega`{=tex}=
```{=tex}
\begin{bmatrix}
X\\
\Upsilon
\end{bmatrix}
```
, `\qquad`{=tex} X'=
```{=tex}
\begin{bmatrix}
A&B
\end{bmatrix}
```
`\Omega`{=tex}. \]

Then estimate the joint operator with low-rank regression/SVD.

## Recommender interpretation

Possible states:

\[ x_k=`\text{user/population preference state}`{=tex} \]

Possible controls:

\[
u_k=`\text{exposure/recommendation policy or content supplied}`{=tex}.
\]

Then:

-   (A): endogenous evolution of preference,
-   (B): effect associated with recommendation/exposure inputs.

## Major causal warning

Logged recommendation exposures are **not automatically valid control
interventions**. If the historical policy selected what users saw, (u_k)
is confounded with user state and policy logic.

Therefore DMDc should not be interpreted causally unless the data design
supports it.

If randomized-exposure data are available, DMDc becomes more
interesting, but careful off-policy/counterfactual reasoning is still
required.

------------------------------------------------------------------------

# 10. Hidden Markov Models (HMMs) --- useful comparison model

The DMD book connects delay-coordinate methods to Markov/state-space
ideas.

An HMM assumes an unobserved state

\[ z_t`\in`{=tex}{1,`\ldots`{=tex},K} \]

with transition probabilities

\[ P(z\_{t+1}=j`\mid `{=tex}z_t=i)=T\_{ij} \]

and observations generated according to

\[ P(y_t`\mid `{=tex}z_t)=E\_{z_t}(y_t). \]

For recommendation, latent states might loosely correspond to behavioral
regimes:

``` text
casual browsing
    ->
topic-focused viewing
    ->
high-engagement session
```

HMMs and DMD represent different hypotheses:

  -----------------------------------------------------------------------
  Method                  State                   Dynamics
  ----------------------- ----------------------- -----------------------
  HMM                     discrete latent state   probabilistic
                                                  transitions

  DMD                     continuous state        linear low-rank
                                                  evolution

  Delay-DMD               continuous              low-rank evolution with
                          history-augmented state memory
  -----------------------------------------------------------------------

A CPU-only HMM is therefore a useful baseline against DMD-derived
temporal features.

------------------------------------------------------------------------

# 11. Extended DMD / Koopman observables --- later-stage nonlinear extension

Classical DMD models dynamics directly in the measured coordinates:

\[ x\_{k+1}`\approx `{=tex}Ax_k. \]

For nonlinear dynamics, define observables

\[ y=g(x) \]

and seek approximately linear dynamics in observable space:

\[ y\_{k+1}`\approx `{=tex}K y_k. \]

Possible recommender observables include:

\[ g(x)= \[ x,; x\^2,; `\log`{=tex}(1+x),; `\text{ratios}`{=tex},;
`\text{cross-features}`{=tex},`\ldots`{=tex}\]. \]

Extended DMD explicitly constructs such nonlinear observables.

Kernel DMD uses kernel methods to represent a richer observable space
implicitly.

## Why it is not first priority

It increases:

-   computational cost,
-   feature-design complexity,
-   overfitting risk,
-   hyperparameter search.

Only test it if linear/Delay-DMD provides evidence that temporal
dynamics are useful but leaves systematic nonlinear residual structure.

------------------------------------------------------------------------

# 12. Koopman viewpoint

The theoretical motivation behind DMD is the Koopman operator.

For nonlinear state dynamics

\[ x\_{k+1}=F(x_k), \]

the Koopman operator (`\mathcal `{=tex}K) acts on measurement functions
(g):

\[ (`\mathcal `{=tex}Kg)(x) = g(F(x)). \]

Although (F) may be nonlinear, (`\mathcal `{=tex}K) is linear in
function space.

DMD can be interpreted as learning a finite-dimensional approximation to
this operator from data.

For the recommender project, the practical lesson is:

> If raw interaction states are not close to linearly evolving,
> carefully chosen observables or embeddings may produce a
> representation whose evolution is easier to approximate linearly.

Do not overclaim that a finite DMD model has discovered the true Koopman
operator.

------------------------------------------------------------------------

# 13. Recommended recommender architecture

A sensible CPU-first architecture is:

``` text
                     interaction log
                           |
             +-------------+-------------+
             |                           |
       static features              temporal aggregation
             |                           |
       user/item/category              x_1 ... x_T
             |                           |
             |                 delay embedding / DMD
             |                           |
             |                   temporal features
             |                           |
             +-------------+-------------+
                           |
                    Factorization Machine
                           |
                     long_view score
                           |
                    GAUC / nDCG@K
```

Do not initially replace the official FM baseline. Augment it.

------------------------------------------------------------------------

# 14. Suggested experiment ladder

## E0 --- Static baseline

Official FM.

Record:

-   GAUC,
-   nDCG,
-   runtime,
-   memory,
-   seed variance.

## E1 --- Simple temporal baseline

Add inexpensive features:

-   recent popularity,
-   rolling counts,
-   exponentially decayed counts,
-   time since last activity,
-   recent category preference.

This determines whether temporal information itself helps.

## E2 --- Classical DMD

Add low-rank DMD features.

Question:

> Does DMD beat simple temporal aggregation?

## E3 --- Delay-DMD

Add (s)-step delay coordinates.

Question:

> Does temporal memory improve beyond first-order DMD?

## E4 --- mrDMD

Separate long/medium/short modes.

Question:

> Are multiple timescales useful?

## E5 --- Sparse-DMD

Select a smaller mode set using an (L_1) penalty.

Question:

> Can noisy modes be removed without losing ranking quality?

## E6 --- Noise-aware DMD

Try stricter rank selection / TLS-style correction if modes are
unstable.

## E7 --- HMM comparison

Compare discrete latent regimes against continuous DMD dynamics.

## E8 --- DMDc

Only if exposure/action inputs are meaningful and the experimental
design justifies them.

## E9 --- Extended/kernel DMD

Only if simpler DMD variants have already produced useful gains.

------------------------------------------------------------------------

# 15. Evaluation discipline

The central scientific comparison should be

\[ `\text{Static FM}`{=tex} \]

vs.

\[ `\text{FM + simple temporal features}`{=tex} \]

vs.

\[ `\text{FM + DMD features}`{=tex}. \]

Without the middle baseline, an improvement from DMD cannot be
distinguished from the simpler fact that **recency/trend information
helps**.

For every experiment record:

``` text
Hypothesis
Data/state definition
Time-window definition
DMD variant
Rank r
Delay depth s (if any)
Features produced
Runtime
Peak memory
GAUC
nDCG
Delta vs FM
Delta vs simple temporal baseline
Stability across seeds/windows
Decision: keep / modify / abandon
```

------------------------------------------------------------------------

# 16. Data leakage rules

Temporal DMD makes leakage especially easy.

For a prediction at time (t), every DMD feature must be computed only
from information available **strictly before or at the permitted
prediction cutoff**.

Never compute modes using validation/test snapshots and then use those
modes as training-time features.

For chronological train/validation/test splits:

``` text
TRAIN ------| VALIDATION ------| TEST
            ^
DMD fitted only on information allowed before prediction
```

If DMD is updated online, reproduce that update protocol identically
during evaluation.

------------------------------------------------------------------------

# 17. Rank selection

The rank (r) is one of the most important DMD hyperparameters.

Too small:

\[ r`\downarrow `{=tex}`\Rightarrow `{=tex}`\text{underfitting}`{=tex}
\]

Too large:

\[
r`\uparrow `{=tex}`\Rightarrow `{=tex}`\text{noise + instability + CPU cost}`{=tex}
\]

Useful approaches:

1.  singular-value elbow,
2.  retained-energy threshold,
3.  validation search over a small rank grid,
4.  noise-aware singular-value thresholding.

For CPU experiments, start with a deliberately small grid such as

\[ r`\in`{=tex}{5,10,20,40}. \]

------------------------------------------------------------------------

# 18. Time-window selection

DMD assumes snapshots correspond to temporal evolution, so the
definition of (x_t) matters as much as the algorithm.

Candidate windows:

-   hourly,
-   6-hour,
-   daily,
-   multi-day.

Too short:

-   very sparse/noisy snapshots.

Too long:

-   temporal dynamics are averaged away.

Treat window width as a first-class hyperparameter.

A useful strategy is to first examine simple temporal autocorrelation /
stability before running DMD.

------------------------------------------------------------------------

# 19. What not to implement initially

## Raw full user-item DMD

Avoid constructing a dense state of all users × all items.

Reasons:

-   memory,
-   sparsity,
-   poor conditioning,
-   expensive SVD,
-   difficult interpretation.

Compress/aggregate first.

## Fluid-dynamics-specific material

The DMD book contains extensive fluid examples. They explain DMD but do
not transfer directly to recommendation.

## Video background/foreground DMD

The fact that the recommendation items are videos does **not** make DMD
video-background separation relevant unless raw video frames are being
analyzed.

## Full kernel DMD

Too expensive/complex for the first CPU experiments.

## DMDc as causal modeling without justification

Exposure is not automatically an intervention.

------------------------------------------------------------------------

# 20. Practical CPU implementation notes

Use NumPy/SciPy-style linear algebra.

Core classical DMD pseudocode:

``` python
# X:  [state_dim, T-1]
# Xp: [state_dim, T-1]

U, s, Vh = svd(X, full_matrices=False)

Ur = U[:, :r]
Sr = diag(s[:r])
Vr = Vh.conj().T[:, :r]

A_tilde = Ur.conj().T @ Xp @ Vr @ inv(Sr)

eigvals, W = eig(A_tilde)

Phi = Xp @ Vr @ inv(Sr) @ W

omega = log(eigvals) / dt
b = pinv(Phi) @ X[:, 0]
```

For larger matrices:

-   use truncated/randomized SVD,
-   use sparse matrices where appropriate,
-   avoid materializing dense user-item matrices,
-   cache temporal aggregates,
-   keep DMD rank small,
-   benchmark DMD preprocessing separately from FM training.

------------------------------------------------------------------------

# 21. Priority summary

  -----------------------------------------------------------------------
  Priority                Method                  Reason
  ----------------------- ----------------------- -----------------------
  1                       Classical low-rank DMD  Necessary baseline;
                                                  cheap; interpretable

  2                       Delay/Hankel DMD        Best fit for behavioral
                                                  memory/sequences

  3                       mrDMD                   Captures
                                                  long/medium/short
                                                  temporal structure

  4                       Sparsity-promoting DMD  Mode selection,
                                                  denoising, CPU
                                                  efficiency

  5                       Noise-aware DMD         Useful if
                                                  modes/eigenvalues are
                                                  unstable

  6                       HMM comparison          Strong CPU-only
                                                  alternative temporal
                                                  hypothesis

  7                       DMDc                    Interesting for
                                                  exposure/input effects,
                                                  but causal caveats

  8                       Extended/kernel DMD     Nonlinear extension
                                                  after simpler methods
                                                  work

  9                       Compressed DMD          Useful mainly if state
                                                  dimensionality becomes
                                                  limiting
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 22. Recommended research question

The cleanest high-level hypothesis is:

> **Can low-rank temporal modes extracted from historical interaction
> data provide ranking signal beyond a static Factorization Machine and
> ordinary recency/popularity features?**

Then refine it:

### H1 --- Temporal signal

\[ `\text{FM + temporal baseline}`{=tex} \> `\text{FM}`{=tex} \]

### H2 --- Dynamical signal

\[ `\text{FM + DMD}`{=tex} \> `\text{FM + temporal baseline}`{=tex} \]

### H3 --- Memory

\[ `\text{FM + Delay-DMD}`{=tex} \> `\text{FM + DMD}`{=tex} \]

### H4 --- Multiple timescales

\[ `\text{FM + mrDMD}`{=tex} \> `\text{FM + Delay-DMD}`{=tex} \]

### H5 --- Sparse dynamics

\[
`\text{Sparse-DMD retains/improves quality at lower complexity.}`{=tex}
\]

This sequence makes the contribution falsifiable. If H1 fails, DMD
should probably be deprioritized. If H1 succeeds but H2 fails, simple
temporal features are sufficient. If H2/H3/H4 succeeds, DMD is
contributing information beyond ordinary recency.

------------------------------------------------------------------------

# 23. Source provenance

This handoff was compiled primarily from:

**J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, Joshua L.
Proctor, *Dynamic Mode Decomposition: Data-Driven Modeling of Complex
Systems*, SIAM, 2016.**

Most relevant portions:

-   Chapter 1 --- classical DMD architecture and algorithm
-   Chapter 5 --- Multiresolution DMD
-   Chapter 6 --- DMD with Control
-   Chapter 7 --- Delay Coordinates, ERA, Hankel matrices, Hidden Markov
    Models
-   Chapter 8 --- noise, rank truncation, DMD spectrum
-   Chapter 9 --- compressed sensing, sparsity-promoting DMD, compressed
    DMD
-   Chapter 10 --- nonlinear observables, extended DMD, kernel DMD

This document is intentionally self-contained so those chapters do not
need to be available to the next model.
