import numpy as np

# 缺失掩码，0表示缺失，1表示非缺失

def mcar_masker(shape = (112, 10000), pi=0.1, seed=42):
    # Missing Completely at Random
    np.random.seed(seed)
    mask = np.random.choice([0, 1], size=shape, p=[pi, 1-pi])
    return mask


def seq_masker(shape = (112, 10000), pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=42):
    # Sequential Missing
    masks = np.zeros(shape, dtype=int)
    for s in range(shape[0]):
        masks[s] = single_seq_masker(
            T=shape[1], pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=seed + s
        )
    return masks


def scm_masker(S_cluster: list, shape=(112, 10000), pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, pi_hat=0.95, seed=42):
    # Spatially Correlated Missing  
    masks = np.zeros(shape, dtype=int)
    for s, cluster in enumerate(S_cluster):
        np.random.seed(seed + s)
        mask_base = np.random.choice([0, 1], size=shape[1], p=[pi_hat, 1-pi_hat])
        masks[s] = mask_base | single_seq_masker(
            T=shape[1], pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=seed + cluster
        )
    return masks


def single_seq_masker(T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=42):
    # single sequential missing
    # 单个缺失块的长度服从binomial(n1, p1)，非缺失块的长度为 L_obse_base+服从binomial(n0, p0)

    n_miss = int(T * pi) # 缺失数
    n_obse = T - n_miss # 非缺失数
    k = int(n_miss // (n1 * p1)) # 缺失块的个数，非缺失块的个数为 k + 1
    n0 = (n_obse // (k + 1) - L_obse_base) / p0 # 缺失块的长度随机变量delta～Binomial(n_0, p_0)
    if n0 < 1:
        raise ValueError("n0 must be greater than or equal to 1")
    
    # 生成k个binomial(n1, p1)、k+1个binomial(n0, p0)
    np.random.seed(seed)
    miss_blocks = np.random.binomial(n1, p1, k)
    obse_blocks = np.random.binomial(n0, p0, k + 1) + L_obse_base

    miss_idx, mask = 0, np.ones(T, dtype=int)
    for i in range(k):
        mask[miss_idx+obse_blocks[i]:miss_idx+obse_blocks[i]+miss_blocks[i]] = 0
        miss_idx += obse_blocks[i] + miss_blocks[i]
    # 随机找一个点分成两块，然后交换这两块
    split_idx = np.random.randint(1, T-1)
    mask = np.concatenate((mask[split_idx:], mask[:split_idx]))
    return mask

def mask_mapping(mask, y): # 将掩码应用到实际值
    return np.where(mask==1, y, np.nan)