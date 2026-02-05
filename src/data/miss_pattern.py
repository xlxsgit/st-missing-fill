import numpy as np


def mask_mcar(S=112, T=10000, pi=0.1, seed=42):
    '''
    生成MCAR缺失掩码

    参数：
    S: 站点总数
    T: 序列总长度
    pi: 目标缺失率
    seed: 随机种子
    
    返回：
    mask: 缺失掩码，0表示缺失，1表示非缺失
    '''
    np.random.seed(seed)
    mask = np.random.choice([0, 1], size=(S, T), p=[1-pi, pi])
    return mask


def mask_seq(S=112, T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=42):
    '''
    生成 S 个站点的 SEQ 缺失掩码，每个站点的缺失模式独立生成
    返回形状：(S, T)
    '''
    masks = np.zeros((S, T), dtype=int)
    for s in range(S):
        masks[s] = mask_single_seq(
            T=T, pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=seed + s
        )
    return masks


def mask_spatial(S_cluster: list, T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, pi_hat=0.95, seed=42):
    '''
    生成 S 个站点的 SPATIAL 缺失掩码，同一个聚类内的站点会同时缺失
    返回形状：(S, T)
    '''
    S = len(S_cluster)
    masks = np.zeros((S, T), dtype=int)
    for s, cluster in enumerate(S_cluster):
        np.random.seed(seed + s)
        mask_base = np.random.choice([0, 1], size=T, p=[1-pi_hat, pi_hat])
        masks[s] = mask_base & mask_single_seq(
            T=T, pi=pi,
            n1=n1, p1=p1,
            L_obse_base=L_obse_base, p0=p0,
            seed=seed + cluster
        )
    return masks


def mask_single_seq(T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=42):
    '''
    生成SEQ缺失掩码，每个缺失块的长度服从参数为n_1, p_1的二项分布，每个非缺失块的长度最短为L_obse_base，服从参数为n_0, p_0的二项分布
    参数：
    T: 序列总长度
    pi: 目标缺失率
    n1: 单个缺失块长度的参数n_1
    p1: 单个缺失块长度的参数p_1
    L_obse_base: 非缺失块的最短长度
    p0: 缺失块的长度随机变量delta~Binomial(n_0, p_0)的参数p_0
    seed: 随机种子
    
    返回：
    mask: 缺失掩码，0表示缺失，1表示非缺失
    '''
    n_miss = int(T * pi) # 缺失数
    n_obse = T - n_miss # 非缺失数
    k = int(n_miss // (n1 * p1)) # 缺失块的个数，非缺失块的个数为 k + 1
    n0 = (n_obse // (k + 1) - L_obse_base) / p0 # 缺失块的长度随机变量delta～Binomial(n_0, p_0)
    
    # 生成k个binomial(n1, p1)、k+1个binomial(n0, p0)
    np.random.seed(seed)
    miss_blocks = np.random.binomial(n1, p1, k)
    obse_blocks = np.random.binomial(n0, p0, k + 1) + L_obse_base

    miss_idx, mask = 0, np.zeros(T, dtype=int)
    for i in range(k):
        mask[miss_idx+obse_blocks[i]:miss_idx+obse_blocks[i]+miss_blocks[i]] = 1
        miss_idx += obse_blocks[i] + miss_blocks[i]
    return mask

