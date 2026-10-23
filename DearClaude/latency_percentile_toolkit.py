"""
Reusable functions for estimating end-to-end latency percentiles from
per-hop p50/p75/p90/p99. See latency_percentile_context.md for the
methodology writeup and references this implements.
"""
import numpy as np
from scipy import stats, optimize

LEVELS = (0.50, 0.75, 0.90, 0.99)
Z = stats.norm.ppf(LEVELS)  # [0, 0.6745, 1.2816, 2.3263]


def fit_lognormal(percentiles, levels=LEVELS):
    """percentiles: sequence of 4 quantile values matching `levels`.
    Returns (mu, sigma, r2). r2 < ~0.95 means this hop isn't well
    described by a single lognormal -- treat tail extrapolation from
    it with suspicion (see fat-tail handling below)."""
    lnq = np.log(percentiles)
    z = stats.norm.ppf(levels)
    A = np.vstack([np.ones_like(z), z]).T
    (mu, sigma), *_ = np.linalg.lstsq(A, lnq, rcond=None)
    pred = A @ [mu, sigma]
    r2 = 1 - np.sum((lnq - pred) ** 2) / np.sum((lnq - lnq.mean()) ** 2)
    return mu, sigma, r2


def lognormal_mean_var(mu, sigma):
    mean = np.exp(mu + sigma**2 / 2)
    var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    return mean, var


def lognormal_quantile(mu, sigma, p):
    return np.exp(mu + sigma * stats.norm.ppf(p))


def naive_sum(hop_percentiles, p_index):
    """hop_percentiles: dict hop_name -> (p50, p75, p90, p99).
    p_index: index into LEVELS, e.g. 3 for p99."""
    return sum(v[p_index] for v in hop_percentiles.values())


def fenton_wilkinson(hop_fits, correlations=None):
    """hop_fits: dict hop_name -> (mu, sigma) from fit_lognormal.
    correlations: optional dict (hop_i, hop_j) -> rho, raw-scale
    correlation between hop latencies. Omit to assume independence.
    Returns (mu_S, sigma_S) for the fitted lognormal of the sum."""
    names = list(hop_fits)
    means, varis = {}, {}
    for n in names:
        means[n], varis[n] = lognormal_mean_var(*hop_fits[n])

    E_S = sum(means.values())
    Var_S = sum(varis.values())
    if correlations:
        for (i, j), rho in correlations.items():
            Var_S += 2 * rho * np.sqrt(varis[i] * varis[j])

    sigma_S2 = np.log(1 + Var_S / E_S**2)
    sigma_S = np.sqrt(sigma_S2)
    mu_S = np.log(E_S) - sigma_S2 / 2
    return mu_S, sigma_S


def dominant_hop(hop_fits):
    """Fat-tail diagnostic: the hop with the largest fitted sigma is
    the one most likely to be the 'single big jump' behind rare
    end-to-end tail events (see context doc, 'single-big-jump' section).
    Flag it if its sigma is a clear outlier vs the rest."""
    return max(hop_fits, key=lambda n: hop_fits[n][1])


def decomposition_estimate(hop_percentiles, dominant, p_index):
    """Fat-tail correction: median(other hops) + dominant hop's own
    measured percentile at p_index. In toy validation this beat both
    naive_sum and fenton_wilkinson at p99 when one hop had a clearly
    fatter tail than the rest."""
    median_idx = LEVELS.index(0.50)
    others = sum(v[median_idx] for n, v in hop_percentiles.items() if n != dominant)
    return others + hop_percentiles[dominant][p_index]


class SyntheticStressTest:
    """Validates the estimators above against known ground truth.

    Generates per-hop latency from a lognormal 'fast path' + rare
    lognormal 'slow path' mixture (simulates retries/cache-misses/GC
    pauses) -- deliberately NOT pure lognormal, to check whether
    fit_lognormal/fenton_wilkinson/decomposition_estimate hold up when
    the real data doesn't match the modeling assumption exactly.

    hop_specs: dict name -> (w_slow, (mu_fast, sigma_fast), (mu_slow, sigma_slow))
    """

    def __init__(self, hop_specs, n=8_000_000, seed=1):
        self.hop_specs = hop_specs
        self.n = n
        rng = np.random.default_rng(seed)
        self.true_samples, self.measured_pct = {}, {}
        for name, (w, fast, slow) in hop_specs.items():
            is_slow = rng.random(n) < w
            x = np.where(is_slow, rng.lognormal(*slow, n), rng.lognormal(*fast, n))
            self.true_samples[name] = x
            self.measured_pct[name] = tuple(np.quantile(x, LEVELS))
        self._total = sum(self.true_samples.values())

    def ground_truth(self, p):
        return np.quantile(self._total, p)

    def single_big_jump(self, p):
        """Which hop is the 'single big jump' behind true tail events at
        percentile p? Returns dict name -> fraction of tail events it
        dominates. See toolkit's dominant_hop() for the cheap proxy
        (largest fitted sigma) that doesn't need this ground truth."""
        threshold = self.ground_truth(p)
        tail_mask = self._total > threshold
        names = list(self.hop_specs)
        typical = np.array([self.true_samples[n][~tail_mask].mean() for n in names])
        stacked = np.vstack([self.true_samples[n][tail_mask] for n in names])
        excess = stacked - typical[:, None]
        dominant_idx = np.argmax(excess, axis=0)
        return {n: float(np.mean(dominant_idx == i)) for i, n in enumerate(names)}

    def tail_probability_matching(self, hop_fits, p):
        """Asymptotic single-big-jump estimator: solve sum_i P(X_i>t) = 1-p
        using each hop's fitted lognormal tail. Included for completeness --
        in practice this underperformed decomposition_estimate at p99
        because lognormal tails converge to the asymptotic regime slowly."""
        def total_exceed_prob(t):
            return sum(stats.lognorm.sf(t, s=s, scale=np.exp(mu)) for mu, s in hop_fits.values())
        return optimize.brentq(lambda t: total_exceed_prob(t) - (1 - p), 1.0, 1e6)

    def compare_estimators(self, p=0.99):
        p_index = LEVELS.index(p)
        truth = self.ground_truth(p)
        naive = naive_sum(self.measured_pct, p_index)

        hop_fits = {n: fit_lognormal(pct)[:2] for n, pct in self.measured_pct.items()}
        mu_S, sigma_S = fenton_wilkinson(hop_fits)
        fw = lognormal_quantile(mu_S, sigma_S, p)

        dom = dominant_hop(hop_fits)
        decomp = decomposition_estimate(self.measured_pct, dom, p_index)
        tail_match = self.tail_probability_matching(hop_fits, p)

        return {
            "truth": truth, "dominant_hop": dom,
            "naive_sum": naive, "fenton_wilkinson": fw,
            "decomposition": decomp, "tail_probability_matching": tail_match,
        }

    def report(self, p=0.99):
        print(f"{'hop':8s} {'mu':>7s} {'sigma':>7s} {'R2':>6s}")
        for name, pct in self.measured_pct.items():
            mu, sigma, r2 = fit_lognormal(pct)
            print(f"{name:8s} {mu:7.4f} {sigma:7.4f} {r2:6.4f}")

        dom_frac = self.single_big_jump(p)
        print(f"\nSingle-big-jump dominance at p{int(p*100)} tail events:")
        for n, frac in sorted(dom_frac.items(), key=lambda kv: -kv[1]):
            print(f"  {n:8s}: {frac*100:5.1f}%")

        res = self.compare_estimators(p)
        print(f"\n{'estimator':28s} {'value':>10s} {'ratio to truth':>15s}")
        for label, key in [
            ("TRUE (ground truth)", "truth"),
            ("naive sum", "naive_sum"),
            ("Fenton-Wilkinson", "fenton_wilkinson"),
            ("tail-probability matching", "tail_probability_matching"),
            (f"median(others)+{res['dominant_hop']}'s own p{int(p*100)}", "decomposition"),
        ]:
            val = res[key]
            ratio = "1.00" if key == "truth" else f"{val / res['truth']:.2f}"
            print(f"{label:28s} {val:10.2f} {ratio:>15s}")


if __name__ == "__main__":
    # Example wiring -- replace with real per-hop p50/p75/p90/p99 (ms).
    hop_percentiles = {
        "gateway": (4.73, 6.02, 7.52, 16.49),
        "auth":    (7.44, 9.48, 11.91, 43.88),
        "db":      (16.73, 23.04, 31.70, 158.48),
        "cache":   (1.92, 2.27, 2.66, 3.65),
        "render":  (9.07, 11.93, 15.39, 34.27),
    }

    hop_fits = {}
    print(f"{'hop':8s} {'mu':>7s} {'sigma':>7s} {'R2':>6s}")
    for name, pct in hop_percentiles.items():
        mu, sigma, r2 = fit_lognormal(pct)
        hop_fits[name] = (mu, sigma)
        flag = "  <- check vs real data" if r2 < 0.95 else ""
        print(f"{name:8s} {mu:7.4f} {sigma:7.4f} {r2:6.4f}{flag}")

    p99_idx = LEVELS.index(0.99)
    naive = naive_sum(hop_percentiles, p99_idx)

    mu_S, sigma_S = fenton_wilkinson(hop_fits)
    fw_p99 = lognormal_quantile(mu_S, sigma_S, 0.99)

    dom = dominant_hop(hop_fits)
    decomp_p99 = decomposition_estimate(hop_percentiles, dom, p99_idx)

    print()
    print(f"dominant (fattest-tail) hop: {dom}")
    print(f"naive sum p99:            {naive:7.2f}")
    print(f"Fenton-Wilkinson p99:     {fw_p99:7.2f}")
    print(f"decomposition estimate:  {decomp_p99:7.2f}")

    # --- Stress test: do the estimators above hold up against KNOWN
    # ground truth when the real distribution isn't pure lognormal? ---
    print("\n" + "=" * 60)
    print("SyntheticStressTest (mixture ground truth, same hop shapes)")
    print("=" * 60)
    hop_specs = {
        "gateway": (0.010, (1.55, 0.35), (3.60, 0.40)),
        "auth":    (0.015, (2.00, 0.35), (3.95, 0.40)),
        "db":      (0.030, (2.80, 0.45), (4.85, 0.50)),  # cache misses -> fattest tail
        "cache":   (0.005, (0.65, 0.25), (2.55, 0.30)),
        "render":  (0.010, (2.20, 0.40), (4.25, 0.40)),
    }
    SyntheticStressTest(hop_specs).report(p=0.99)
