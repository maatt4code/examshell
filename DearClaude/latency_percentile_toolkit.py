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
