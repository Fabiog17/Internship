"""
Contact map noising — due modalità selezionabili con USE_METROPOLIS:

  True  → RFIM Metropolis: rumore spazialmente correlato via sweep Ising 2D.
           A T = 0.9·T_c i cluster di spin sono grandi e compatti, simili agli
           errori DCA cluster-shaped.

  False → indipendente per sito: ogni coppia (i,j) riceve un falso positivo con
           probabilità Bernoulli ~ alpha · p(|i-j|), dove p(d) è la densità di
           contatti veri a separazione d nella stessa proteina.  Equivalente alla
           statistica di sito simmetrizzata usata dall'intern, senza correlazioni
           spaziali.

Entrambe supportano falsi negativi opzionali (fn_p0).
"""

import numpy as np

# 2D Ising critical temperature (Onsager, J=1): T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


# ── Helpers condivisi ─────────────────────────────────────────────────────────

def _contact_density_by_offset(true_map: np.ndarray) -> np.ndarray:
    """p(d) = frazione di contatti veri a separazione d = |i-j|."""
    L = true_map.shape[0]
    idx = np.arange(L)
    D = np.abs(idx[:, None] - idx[None, :])
    p_d = np.zeros(L)
    for d in range(1, L):
        mask = D == d
        if mask.sum() > 0:
            p_d[d] = true_map[mask].mean()
    return p_d


def _apply_fn(true_map: np.ndarray, fn_p0: float, fn_decay: float, rng) -> np.ndarray:
    """Rimuove contatti veri con probabilità fn_p0 * exp(-|i-j| / fn_decay)."""
    L = true_map.shape[0]
    idx = np.arange(L)
    D = np.abs(idx[:, None] - idx[None, :])
    p_fn = fn_p0 * np.exp(-D / fn_decay)
    drop = np.triu(rng.random((L, L)) < p_fn, k=1)
    drop |= drop.T
    return np.where(drop, np.int8(0), true_map)


# ── Metropolis RFIM ───────────────────────────────────────────────────────────

def _extract_sites(L: int, min_sep: int = 2):
    sites = [(i, j) for i in range(L) for j in range(i + min_sep, L)]
    site_to_idx = {s: k for k, s in enumerate(sites)}
    neighbors = []
    for (i, j) in sites:
        nbrs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            a, b = (ni, nj) if ni < nj else (nj, ni)
            if a != b and (a, b) in site_to_idx:
                nbrs.append(site_to_idx[(a, b)])
        neighbors.append(np.array(nbrs, dtype=np.int64))
    return sites, site_to_idx, neighbors


def _field_from_p_d(sites, p_d, alpha: float, T_seed: float = 1.0, eps: float = 1e-3):
    H = np.empty(len(sites))
    for k, (i, j) in enumerate(sites):
        p = np.clip(alpha * p_d[abs(i - j)], eps, 1 - eps)
        H[k] = 0.5 * T_seed * np.log(p / (1 - p))
    return H


def _init_from_field(H, T_seed: float, rng):
    p = 1.0 / (1.0 + np.exp(-2.0 * H / T_seed))
    return np.where(rng.random(len(H)) < p, 1.0, -1.0)


def _metropolis_sweep(S, H, neighbors, J: float, T: float, rng):
    for k in rng.permutation(len(S)):
        s = S[k]
        nb = neighbors[k]
        N = S[nb].sum() if len(nb) else 0.0
        dE = 2.0 * s * (J * N + H[k])
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            S[k] = -s
    return S


def build_noisy_map_metropolis(
    true_map: np.ndarray,
    alpha: float = 0.5,
    J: float = 1.0,
    T: float = 0.9 * TC,
    n_sweeps: int = 30,
    min_sep: int = 2,
    fn_p0: float = 0.0,
    fn_decay: float = 6.0,
    seed: int = 0,
    zero_field: bool = True,
) -> np.ndarray:
    """
    Rumore correlato spazialmente via Metropolis RFIM.

    alpha     : scala del campo di bias (densità iniziale dei semi).
                Ignorato se zero_field=True.
    J         : coupling ferromagnetico (più alto → cluster più grandi)
    T         : temperatura; default 0.9·T_c (fase ordinata, cluster grandi)
    n_sweeps  : sweep Metropolis
    zero_field: True → campo H=0, puro Ising senza bias verso la densità vera
    """
    rng = np.random.default_rng(seed)
    L = true_map.shape[0]

    sites, _, neighbors = _extract_sites(L, min_sep=min_sep)
    p_d = _contact_density_by_offset(true_map)
    H = _field_from_p_d(sites, p_d, alpha=alpha)
    S = _init_from_field(H, T_seed=1.0, rng=rng)

    if zero_field:
        H = np.zeros_like(H)

    for _ in range(n_sweeps):
        _metropolis_sweep(S, H, neighbors, J, T, rng)

    noise_field = np.zeros((L, L), dtype=np.int8)
    for k, (i, j) in enumerate(sites):
        if S[k] > 0:
            noise_field[i, j] = noise_field[j, i] = 1

    tm = _apply_fn(true_map, fn_p0, fn_decay, rng) if fn_p0 > 0.0 else true_map
    noisy = np.maximum(noise_field, tm)
    np.fill_diagonal(noisy, 0)
    return noisy


# ── Indipendente per sito ─────────────────────────────────────────────────────

def build_noisy_map_independent(
    true_map: np.ndarray,
    alpha: float = 0.5,
    min_sep: int = 2,
    fn_p0: float = 0.0,
    fn_decay: float = 6.0,
    seed: int = 0,
    **_ignored,          # ignora J, T, n_sweeps se passati per sbaglio
) -> np.ndarray:
    """
    Rumore indipendente per coppia basato sulla statistica di sito simmetrizzata.

    Ogni coppia (i,j) con j-i >= min_sep riceve un falso positivo con
    probabilità Bernoulli:  p_FP(i,j) = alpha · p(|i-j|)
    dove p(d) è la densità di contatti veri a separazione d nella stessa proteina.

    Nessuna correlazione spaziale — equivalente all'approccio dell'intern
    (campionamento per-coppia indipendente).
    """
    rng = np.random.default_rng(seed)
    L = true_map.shape[0]
    p_d = _contact_density_by_offset(true_map)

    noise_field = np.zeros((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(i + min_sep, L):
            p = np.clip(alpha * p_d[j - i], 0.0, 1.0)
            if rng.random() < p:
                noise_field[i, j] = noise_field[j, i] = 1

    tm = _apply_fn(true_map, fn_p0, fn_decay, rng) if fn_p0 > 0.0 else true_map
    noisy = np.maximum(noise_field, tm)
    np.fill_diagonal(noisy, 0)
    return noisy


# ── Entry point unico ─────────────────────────────────────────────────────────

def build_noisy_map(
    true_map: np.ndarray,
    use_metropolis: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Dispatcher: chiama build_noisy_map_metropolis o build_noisy_map_independent
    in base a use_metropolis.

    Tutti i kwargs vengono inoltrati alla funzione scelta.
    """
    if use_metropolis:
        return build_noisy_map_metropolis(true_map, **kwargs)
    return build_noisy_map_independent(true_map, **kwargs)
