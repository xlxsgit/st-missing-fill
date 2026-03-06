# 基于 Transformer 超网络与双向时空对齐的非迭代变系数风流模型

根据统计学时间序列分析（尤其是针对历史数据的插补而非实时预测），仅仅依赖历史信息的单向自回归是不充分的。风场和气象特征具有高度的时空平滑性，目标时刻 $t$ 的状态不仅受过去 $t-1, t-2$ 影响，也同样受未来 $t+1, t+2$ 制约（因为未来的状态是当前状态的因果延续）。

我们将框架升级为**双向自回归 (Bidirectional STAR)**，同时彻底解剖那个充当“神之手”的 Transformer 超网络到底长什么样、是如何同时输出初值和系数的。

---

## 1. 终极非迭代：双向定常统计主方程

我们引入一个中心化滞后窗口 $l \in [-L, L] \setminus \{0\}$，即前后各看 $L$ 步。单次前向输出的最终插值 $\hat{Y}_{s,t}$ 等式为：

$$
\begin{align*}
\hat{Y}_{s,t} = \ &\sum_{l \in [-L, L], l \neq 0} \mathbf{\color{red}{A}_{s,l}(t)} \cdot \tilde{Y}_{s,t+l}^* \quad &\text{[本地时间双向平滑项]} \\ 
&+ \sum_{l \in [-L, L], l \neq 0} \mathbf{\color{red}{\beta}_{s,l}(t)} \cdot \left[ \sum_{s' \neq s}^S \Omega\Big(l, \tau_{s',s}\Big) \cdot \tilde{\alpha}_{s',s} \cdot \tilde{Y}_{s',t+l}^* \right] \quad &\text{[风流动对齐空间交互项]} \\
&+ \sum_{p=1}^P \mathbf{\color{red}{B}_{s,p}(t)} \cdot X_{s,t}^{(p)} \quad &\text{[局地微气候驱动项]}
\end{align*}
$$

为了公式的书写优雅与代码实现的高效化，我们定义了**“混合洗底值” (Hybrid Clean Value)** $\tilde{Y}^*$：
$$ \tilde{Y}_{s, k}^* = M_{s, k} \cdot Y_{s, k}^{\text{True}} + (1 - M_{s, k}) \cdot \tilde{Y}_{s, k}^{\text{Transformer}} $$
*语义：如果在时空点 $(s, k)$ 有真实观测风速 ($M=1$)，那 $\tilde{Y}^*$ 就是坚若磐石的真理值。如果该点碰巧缺失 ($M=0$)，它就采用 Transformer 利用全局注意力网络算出来的垫底值 $\tilde{Y}^{\text{Transformer}}$。*

---

## 2. Transformer 超网络的大解剖 (The Anatomy of the Hyper-Network)

这个网络是如何在“端到端（End-to-End）”的一次计算中，既吐出了上面方程中那些红色的极其聪明的系数 $A_l(t), \beta_l(t), B_p(t)$，又顺道吐出了清洗垫底值 $\tilde{Y}^{\text{Transformer}}$ 的？

### 2.1 高维张量输入层 (The Input Constructor)
对于一个包含了 $S$ 个站点、$T$ 个连续时间步的训练/推断窗口，我们拼装出一个三维张量 $\mathbf{X}_{\text{in}} \in \mathbb{R}^{S \times T \times D_{\text{in}}}$。它的特征维度 $D_{\text{in}}$ 包含：
- $\mathbf{Y}_{\text{raw}}$: 带有 NaN 的原始风速（NaN用 0 填充）。
- $\mathbf{M}$: 0/1 的风速缺失掩膜（非常关键，它告诉网络哪里是假数据）。
- $\mathbf{X}_{\text{meteo}}$: 温度、气压梯度、湿度、日照时间等完整协变量。
- $\mathbf{Z}_{\text{topo}}$: 扩展至 $T$ 长度的静态地形特征（LV95坐标、坡度、坡向、海拔、TPI）。
- $\mathbf{\Theta}$: 顺带把已知风向特征也塞进去。

### 2.2 时空编码器 (Spatio-Temporal Transformer Encoder)
在输入 Transformer 前，将 $\mathbf{X}_{\text{in}}$ 通过一个双层线性感知机 (MLP) 升维映射到隐空间大小 $D_{\text{model}}$ (比如 128 或 256)。
再加上 `时间位置编码 (Time Positional Encoding)` 和 `空间图嵌入 (Spatial Embedding)`。

接着送入多层（例如 3 到 6 层）标准 Transformer Encoder Layer，核心为 **Multi-Head Self-Attention**：
$$ \mathbf{H} = \text{LayerNorm}\Big(\mathbf{Z} + \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\Big) $$
这里的 $\mathbf{H} \in \mathbb{R}^{S \times T \times D_{\text{model}}}$ 是最核心的高级隐表示。由于自注意力的特性，如果某个站点 $s$ 在时刻 $t$ 的 $M=0$，它生成的 Query 会疯狂地去吸取周边拥有 $M=1$ 且气压差类似的时间片/站点的特征。

### 2.3 分布式的解耦预测头 (Decoupled Output Heads)
拿到了全域融合好的大张量 $\mathbf{H}$ 后，我们在最后接上三个**完全并行的全连接投影头 (Linear Projection Heads)**，每个头的任务截然不同：

**A. 洗底初值预测头 (Clean Filler Head):**
用来负责把周围真实的风速投射回来，填满假洞。
$$ \tilde{\mathbf{Y}}^{\text{Transformer}} = \text{Linear}_{D_{\text{model}} \to 1}(\mathbf{H}) $$

**B. 时间系数预测头 (Temporal AR Head):**
用来输出前后各 $L$ 步的双向自回归系数（共 $2L$ 个数值）。
$$ \mathbf{\color{red}{A}}(t) = \text{Linear}_{D_{\text{model}} \to 2L}(\mathbf{H}) $$

**C. 辅助系数预测头 (Spatial & Covariate Head):**
用来输出 $2L$ 个空间平流放大权重 $\color{red}{\beta}(t)$ 和 $P$ 个本地微环境驱使系数 $\color{red}{B}(t)$。
$$ [\mathbf{\color{red}{\beta}}(t), \ \mathbf{\color{red}{B}}(t)] = \text{Linear}_{D_{\text{model}} \to 2L + P}(\mathbf{H}) $$

---

## 3. 把它们拼接到一起的前向流 (Forward Flow Execution)

在代码里（比如 PyTorch 的 `forward` 函数）发生的事情极其优雅：

1. `H = Transformer(X_in)`
2. `Y_clean = Head_Filler(H)`
3. 按照 $M$ 矩阵拼装出混合输入矩阵 $\tilde{Y}^* = M_{s,k} Y_{\text{True}} + (1-M_{s,k}) Y_{\text{clean}}$
4. `A, beta, B = Head_Coeffs(H)`
5. 提取预先算好的纯物理公式（距离、高山阻滞、10分钟平均风的流速传导耗时）：$\Omega \cdot \tilde{\alpha}$ 。
6. **One-Shot Execute**: 一把梭哈，把第三步准备好的 $\tilde{Y}^*$ 以及物理张量分别与第四步的系数相乘并累加（本质上是大规模稀疏向量点积）。
7. **获得最终精炼输出**：得到最终的 $\hat{Y}$。

## 4. 这套网络到底在干什么？
你可以把这个 Transformer 视为一个**“统筹全局的特级气象调度员”**。
- 它看了全盘的掩码 $\mathbf{M}$，如果它发现站点 $S_1$ 今天从早到晚都没有数据（全天断网）。
- **调度动作1**：它指挥 `Head_Filler` 根据隔壁山头 $S_2$ 的大风和气压记录估算出一个勉强能用的假值系列给 $S_1$。
- **调度动作2 (最聪明的地方)**：在吐出统计方程系数组合时，它特意把 $S_1$ 自己明天和昨天的时间系数 $\color{red}{A}(t\pm L)$ 压扁至极低。因为它知道那些 $S_1$ 自己的前后数据也都是刚才顺手捏出来的！
- **调度动作3**：同时，它把 $\color{red}{\beta}$ （来自北边 $S_2$ 的物理风传递）和由于今天降水骤降的协变量系数 $\color{red}{B_{\text{precip}}}$ **极大增权**。

这套严密的逻辑通过端对端的 Loss 反向传播进行联合优化，让“黑盒的庞大深度抽象”完全服务于“白盒的精美统计多项式”。
