import numpy as np


def mask_mcar(T=10000, pi=0.1, seed=42):
    '''
    生成MCAR缺失掩码

    参数：
    T: 序列总长度
    pi: 目标缺失率
    seed: 随机种子
    
    返回：
    mask: 缺失掩码，0表示缺失，1表示非缺失
    '''
    np.random.seed(seed)
    mask = np.random.choice([0, 1], size=T, p=[1-pi, pi])
    return mask

def mask_seq(T=10000, pi=0.1, n1=24, p1=0.5, L_obse_base=10, p0=0.5, seed=42):
    '''
    生成SEQ缺失掩码
    
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