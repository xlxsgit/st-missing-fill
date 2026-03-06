import numpy as np

# 缺失掩码，0表示缺失，1表示非缺失

def _validate_shape_and_pi(shape, pi):
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D (stations, timesteps), got {shape}")
    if not (0 < pi < 1):
        raise ValueError(f"pi must be in (0, 1), got {pi}")


def _validate_seq_params(shape, pi, n1, p1, L_obse_base, p0):
    _validate_shape_and_pi(shape, pi)
    if n1 <= 0:
        raise ValueError(f"n1 must be > 0, got {n1}")
    if not (0 < p1 <= 1):
        raise ValueError(f"p1 must be in (0, 1], got {p1}")
    if L_obse_base < 0:
        raise ValueError(f"L_obse_base must be >= 0, got {L_obse_base}")
    if not (0 < p0 <= 1):
        raise ValueError(f"p0 must be in (0, 1], got {p0}")

    T = shape[1]
    n_miss = int(T * pi)
    k = max(1, int(n_miss // (n1 * p1)))

    n_obse = T - n_miss
    
    # Auto-adjust L_obse_base for high missing rates
    max_L = int(n_obse // (k + 1) - p0)
    actual_L_obse = min(L_obse_base, max(0, max_L))
    
    n0 = (n_obse // (k + 1) - actual_L_obse) / p0
    if n0 < 1:
        pass # Let it bypass validation if mathematically squeezed


def _seed_for_index(base_seed, idx):
    # base_seed=None means non-deterministic run-level randomness.
    if base_seed is None:
        return None
    return int(base_seed) + int(idx)


def mcar_masker(shape=(112, 10000), pi=0.1, seed=None):
    # Missing Completely at Random
    _validate_shape_and_pi(shape, pi)
    rng = np.random.default_rng(seed)
    mask = rng.choice([0, 1], size=shape, p=[pi, 1 - pi])
    return mask


def seq_masker(shape=(112, 10000), pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=None):
    # Sequential Missing
    _validate_seq_params(shape, pi, n1, p1, L_obse_base, p0)
    masks = np.zeros(shape, dtype=int)
    for s in range(shape[0]):
        masks[s] = single_seq_masker(
            T=shape[1],
            pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=_seed_for_index(seed, s),
        )
    return masks


def scm_masker(
    S_cluster: list,
    shape=(112, 10000),
    pi=0.1,
    n1=24,
    p1=0.5,
    L_obse_base=10,
    p0=0.5,
    pi_hat=0.95,
    seed=None,
):
    # Spatially Correlated Missing
    _validate_seq_params(shape, pi, n1, p1, L_obse_base, p0)
    if S_cluster is None or len(S_cluster) == 0:
        raise ValueError("S_cluster is required for 'scm' missing pattern")
    if len(S_cluster) != shape[0]:
        raise ValueError(
            f"S_cluster length must equal number of stations ({shape[0]}), got {len(S_cluster)}"
        )
    if not (0 <= pi_hat <= 1):
        raise ValueError(f"pi_hat must be in [0, 1], got {pi_hat}")

    masks = np.zeros(shape, dtype=int)
    for s, cluster in enumerate(S_cluster):
        rng = np.random.default_rng(_seed_for_index(seed, s))
        mask_base = rng.choice([0, 1], size=shape[1], p=[pi_hat, 1 - pi_hat])
        masks[s] = mask_base | single_seq_masker(
            T=shape[1], pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=_seed_for_index(seed, cluster),
        )
    return masks


def single_seq_masker(T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=None):
    # single sequential missing
    # 单个缺失块的长度服从binomial(n1, p1)，非缺失块的长度为 L_obse_base+服从binomial(n0, p0)

    n_miss = int(T * pi)  # 缺失数
    n_obse = T - n_miss  # 非缺失数
    k = max(1, int(n_miss // (n1 * p1)))  # 缺失块的个数，非缺失块的个数为 k + 1
    
    # Fix n0 < 1 by bounding L_obse_base
    max_L = int(n_obse // (k + 1) - p0)
    L_obse_base = min(L_obse_base, max(0, max_L))
    
    n0 = max(1, (n_obse // (k + 1) - L_obse_base) / p0)  # 缺失块的长度随机变量delta～Binomial(n_0, p_0)


    # 生成k个binomial(n1, p1)、k+1个binomial(n0, p0)
    rng = np.random.default_rng(seed)
    miss_blocks = rng.binomial(n1, p1, k)
    obse_blocks = rng.binomial(n0, p0, k + 1) + L_obse_base

    miss_idx, mask = 0, np.ones(T, dtype=int)
    for i in range(k):
        mask[miss_idx + obse_blocks[i] : miss_idx + obse_blocks[i] + miss_blocks[i]] = 0
        miss_idx += obse_blocks[i] + miss_blocks[i]
    # 随机找一个点分成两块，然后交换这两块
    split_idx = rng.integers(1, T - 1)
    mask = np.concatenate((mask[split_idx:], mask[:split_idx]))
    return mask


def mask_mapping(mask, y):  # 将掩码应用到实际值
    return np.where(mask == 1, y, np.nan)


def simulate_missingness(
    ground_y,
    pi,
    miss_pattern,
    S_cluster=None,
    seed=None,
    seq_params=None,
    scm_params=None,
):
    seq_params = seq_params or {}
    scm_params = scm_params or {}

    if miss_pattern == "mcar":
        mask = mcar_masker(ground_y.shape, pi, seed=seed)
    elif miss_pattern == "seq":
        mask = seq_masker(ground_y.shape, pi, seed=seed, **seq_params)
    elif miss_pattern == "scm":
        mask = scm_masker(S_cluster, ground_y.shape, pi, seed=seed, **scm_params)
    else:
        raise ValueError(f"Unknown missing pattern: {miss_pattern}")
    y_pattern = mask_mapping(ground_y, mask)
    y_pattern = y_pattern[..., np.newaxis]
    return y_pattern, mask
