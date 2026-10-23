# Context: Inferring end-to-end latency percentiles from per-hop percentiles

Hi — context below on a latency analysis problem I've been working through. Picking this up where I left off; the data lives on our side, this doc is the methodology/reasoning to work from.

## Problem statement

For each hop in a multi-hop request path, we have 30s-sampled **p50, p75, p90, p99** latency (no raw per-request traces, no joint data across hops for the same request). Goal: estimate **end-to-end** latency percentiles (especially p99) from these per-hop marginal percentiles.

The naive approach — just summing each hop's p99 — is what we're trying to improve on. It implicitly assumes every hop is simultaneously at its own 99th-percentile value on the same request, which is far rarer than the actual 1%-tail event of the sum (under independence, that joint event has probability ~0.01ⁿ, not 0.01). So naive summing is generally a **loose overestimate**, and the degree of overestimation depends on cross-hop correlation — which we can't observe from marginals alone. In the limit of perfectly correlated hops (always slow together), naive summing becomes exact.

## Log-normal assumption + goodness-of-fit check

Latency distributions are typically well-modeled as log-normal (multiplicative jitter effects → normal in log-space). Per hop, fit `(μ, σ)` from the 4 known percentiles via linear regression:

```
ln(Q(p)) = μ + σ·z(p),   z(p) = Φ⁻¹(p)
z(0.50)=0, z(0.75)=0.6745, z(0.90)=1.2816, z(0.99)=2.3263
```

Regress `ln(latency)` on `z(p)` across the 4 points per hop → slope = σ, intercept = μ. The **R² of that regression is the diagnostic**: if a hop's 4 points don't fall on a line in this space, it isn't well-described by a single log-normal, and any tail extrapolation from that fit should be trusted less. In a toy validation (below), the two hops with a real underlying tail-mixture (occasional retries/cache-misses) had visibly lower R² (~0.90–0.92) than the clean hops (~0.96–0.998) — R² below ~0.95 or so is a flag worth checking against raw data if possible.

## Methodology: aggregating across hops

1. **Naive sum** — baseline / upper-bound-ish reference, sum each hop's percentile value directly. Known to overestimate under independence; not a strict mathematical bound in all cases (quantiles/VaR are not generally subadditive), but overestimates in essentially all practical latency cases.
2. **Fenton–Wilkinson moment matching** — fit each hop to log-normal, convert to moments, sum the moments (assuming independence: `Var[S] = ΣVar[Xᵢ]`; add `2Σρᵢⱼσᵢσⱼ` if modeling correlation), then moment-match the sum back to a single log-normal:
   ```
   E[X] = exp(μ+σ²/2),  Var[X] = (exp(σ²)-1)·exp(2μ+σ²)
   σ_S² = ln(1 + Var[S]/E[S]²),  μ_S = ln(E[S]) - σ_S²/2
   Q_S(p) = exp(μ_S + σ_S·z(p))
   ```
3. **Monte Carlo cross-check** — sample from each hop's fitted log-normal, sum, take empirical quantile. Useful to validate step 2's arithmetic, but note: if it samples from the *same fitted* log-normals as step 2, it only checks the FW math, not reality.

### Toy validation (synthetic ground truth, 5 hops, independence assumed)

Built true per-hop generators as log-normal-"fast path" + rare log-normal-"slow path" mixtures (retries/cache-misses/GC), sampled millions of rows, and compared against real ground truth (direct sum of true samples):

| pct | TRUE | naive sum | FW analytic | ratio naive/TRUE | ratio FW/TRUE |
|---|---|---|---|---|---|
| p50 | 42.20 | 39.89 | 39.43 | 0.95 | 0.93 |
| p75 | 50.31 | 52.74 | 57.62 | 1.05 | 1.15 |
| p90 | 62.62 | 69.18 | 81.07 | 1.10 | 1.29 |
| p99 | 188.13 | 256.77 | 145.90 | 1.36 | 0.78 |

Takeaway: naive sum overestimates p99 (+36%), but FW *underestimates* it (-22%) here — because the true per-hop tails are fatter than log-normal, which FW's 4-point fit can't see. **FW only fixes the independence/aggregation error, not per-hop distributional misspecification.** Both errors matter and point in opposite directions.

## Fat-tail handling

Two extra things worth running when a hop's R² looks shaky or its fitted σ is an outlier vs. the other hops:

1. **Single-big-jump diagnostic**: for subexponential tails (log-normal included), rare extreme totals are typically driven by *one* hop having a bad event, not all hops simultaneously. In the toy test, the hop with by far the largest fitted σ caused 96% of the true p99-exceeding events. Practical implication: **identify the hop with the largest fitted σ** — that's your tail-dominating hop, visible straight from the per-hop fits, no joint data needed.
2. **Decomposition estimator**: once the dominant hop is identified, estimate end-to-end p99 as:
   ```
   (sum of MEDIANS of the other hops) + (dominant hop's own measured p99)
   ```
   In the toy test this gave a ratio of **0.96** to ground truth — notably better than both naive sum (1.36) and FW (0.78) — because it uses the dominant hop's *actual measured* p99 rather than an extrapolated log-normal tail.
3. A theoretically "proper" asymptotic method (matching `Σᵢ P(Xᵢ>t) = 0.01` via each hop's fitted tail) actually did *worse* (ratio 0.67) in the toy test — log-normal tails converge to their asymptotic single-big-jump regime very slowly, so p99 usually isn't "far enough" out for this to help. **Don't reach for this over the simpler decomposition estimator without deeper tail data.**
4. The rigorous fix for genuinely fat (power-law) tails is **Extreme Value Theory / Peaks-over-Threshold**: fit a Generalized Pareto Distribution to each hop's tail above a threshold, then combine via FFT-based numerical convolution. **This needs more tail resolution than p50/p75/p90/p99 gives** — ideally raw samples above a threshold, or at least a couple of deeper percentiles (p99.9, p99.99). Worth considering an HDR Histogram–style instrumentation change if we want to do this properly rather than just the decomposition heuristic.

## Open items to resolve once we have the real data

- Compute the regression R² per real hop — flag any hop below ~0.95 as a log-normal-fit risk.
- Identify which hop(s), if any, have a clear outlier σ — candidate for the single-big-jump decomposition estimator.
- If any joint/traced data exists for even a subset of requests, use it to sanity-check the independence assumption (correlation between hops matters a lot — naive sum becomes *exact*, not just an overestimate, if hops are highly correlated).
- Decide whether it's worth capturing deeper tail percentiles (p99.9+) to enable a real GPD fit for the worst hop(s).

## References

**Sum-of-log-normals approximation (Fenton–Wilkinson and alternatives):**
- L.F. Fenton, "The Sum of Log-Normal Probability Distributions in Scatter Transmission Systems," *IRE Trans. Communications Systems*, 1960.
- S. Schwartz, Y.S. Yeh, "On the Distribution Function and Moments of Power Sums with Log-Normal Components," *IEEE Trans. Communications*, 1982.
- N.C. Beaulieu, A.A. Abu-Dayya, P.J. McLane, "Estimating the Distribution of a Sum of Independent Lognormal Random Variables," *IEEE Trans. Communications*, 1995.
- N.B. Mehta, J. Wu, A.F. Molisch, J. Zhang, "Approximating a Sum of Random Variables with a Lognormal," *IEEE Trans. Wireless Communications*, 2007.

**Why summing percentiles isn't a valid general bound (quantile/VaR non-subadditivity):**
- P. Artzner, F. Delbaen, J.-M. Eber, D. Heath, "Coherent Measures of Risk," *Mathematical Finance*, 1999.
- P. Embrechts, A. McNeil, D. Straumann, "Correlation and Dependence in Risk Management: Properties and Pitfalls," 2002.
- J. Dhaene, M. Denuit, M.J. Goovaerts, R. Kaas, D. Vyncke, "The Concept of Comonotonicity in Actuarial Science and Finance," *Insurance: Mathematics and Economics*, 2002.

**Extreme Value Theory / fat tails:**
- A.J. McNeil, R. Frey, P. Embrechts, *Quantitative Risk Management*, Ch. 5 (EVT).
- S. Coles, *An Introduction to Statistical Modeling of Extreme Values*, 2001.
- P. Embrechts, C. Klüppelberg, T. Mikosch, *Modelling Extremal Events for Insurance and Finance*, 1997 (source of the single-big-jump / subexponentiality theory).

**Practical systems-side framing:**
- J. Dean, L.A. Barroso, "The Tail at Scale," *Communications of the ACM*, 2013.
- Gil Tene, "How NOT to Measure Latency" (Strange Loop / QCon talk) — covers coordinated omission and other percentile-measurement pitfalls.
- Google SRE book, SLI/SLO chapter.
