# A finite-sample bound for reliability-weighted edge voting

This note isolates the simplest setting in which weighting can be analyzed. It is not a theorem that bootstrap stability always identifies the correct graph.

## Setup

Fix one directed edge and let its unknown truth be \(y\in\{-1,+1\}\). Local estimator \(k\) returns \(Y_k\in\{-1,+1\}\). Assume conditional independence and

\[
\Pr(Y_k=y)=q_k>\tfrac12.
\]

For deterministic nonnegative weights \(a_k\), predict with

\[
\widehat y=\operatorname{sign}\left(\sum_{k=1}^m a_kY_k\right).
\]

## Proposition

Under the assumptions above,

\[
\Pr(\widehat y\ne y)
\le
\exp\left[
-\frac{
\left(\sum_{k=1}^m a_k(2q_k-1)\right)^2
}{2\sum_{k=1}^m a_k^2}
\right].
\]

## Proof

Multiply all votes by \(y\), so a correct vote is \(+1\). Define \(Z_k=a_kyY_k\). Then \(Z_k\in[-a_k,a_k]\) and

\[
\mathbb E Z_k=a_k(2q_k-1).
\]

An error occurs when \(\sum_k Z_k\le0\). Let \(\mu=\sum_k\mathbb E Z_k\). Hoeffding's inequality gives

\[
\Pr\left(\sum_k(Z_k-\mathbb EZ_k)\le-\mu\right)
\le
\exp\left(-\frac{2\mu^2}{\sum_k(2a_k)^2}\right),
\]

which simplifies to the stated result.

## What the proposition does and does not establish

The bound improves when weights concentrate on estimators with larger accuracy margins \(2q_k-1\), but overly concentrated weights reduce the effective number of voters through \(\sum a_k^2\). Therefore, a useful weight must balance reliability and diversity.

The implementation uses \(a_k=\exp(-\lambda d_k)\), where \(d_k\) is bootstrap disagreement. Connecting the proposition to the algorithm requires an additional calibration assumption: lower bootstrap disagreement must predict larger \(q_k\). Testing that relationship is a central empirical objective, not an established fact.

The independence assumption is also unrealistic when local subgraphs overlap. A stronger analysis should replace Hoeffding with a dependency-graph or martingale bound and make overlap explicit.

