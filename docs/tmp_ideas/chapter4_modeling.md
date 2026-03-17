# 4.2 PG-STAT 物理建模与超网络框架

## 4.2.1 总体算法架构
PG-STAT（Physics‑Guided Spatio‑Temporal Transformer）由三层结构组成：
1. **物理先验层**：计算空间对齐权重 $\tilde{\alpha}_{s',s}(t)$ 与时间对齐核 $\Omega(l,\tau_{s',s}(t))$。
2. **Transformer 超网络**：以完整的观测张量 $\mathbf{Y},\mathbf{R},\Theta,\Lambda,Z,\mathbf{X}$ 为输入，输出动态系数矩阵 $A_{s,l}(t),\beta^{\text{sp}}_{s,l}(t),\gamma_s(t),B_{s,p}(t)$。
3. **插补方程求解层**：将物理先验与动态系数代入时空自回归方程，得到缺失位置的预测值。

## 4.2.2 空间因果对齐机制
- **方位偏差**
$$
\Delta\theta_{s',s}(t)=\min\big(|\Theta_{s',t}-\Phi_{s',s}|,\;360^{\circ}-|\Theta_{s',t}-\Phi_{s',s}|\big)
$$
- **迎风增益**（基于坡向 $\Lambda_s$）
$$
\mu_{s',s}(t)=\max\big(0,\cos\big(\tfrac{\pi}{180^{\circ}}|\Theta_{s',t}-\Lambda_s|\big)\big)
$$
- **综合空间对齐权重**
$$
\tilde{\alpha}_{s',s}(t)=
\begin{cases}
\cos\big(\Delta\theta_{s',s}(t)\big)\,\exp\big(-\tfrac{D_{s',s}}{\sigma_d}\big)\,\exp\big(-\tfrac{|Z_s-Z_{s'}|}{\sigma_z}\big)\,(1+\lambda_{\text{topo}}\mu_{s',s}(t)), & \Delta\theta_{s',s}\le 90^{\circ}\\[4pt]
0, & \Delta\theta_{s',s}>90^{\circ}
\end{cases}
$$

## 4.2.3 时间因果对齐机制
- **有效平流速度**
$$
 v^{\text{eff}}_{s',s}(t)=\max\big(y_{s',t}\cos\Delta\theta_{s',s}(t),\;\varepsilon_v\big)
$$
- **物理时延**
$$
\tau_{s',s}(t)=\frac{D_{s',s}}{T_{\text{gap}}\,v^{\text{eff}}_{s',s}(t)}
$$
- **高斯对齐核**
$$
\Omega\big(l,\tau_{s',s}(t)\big)=\exp\Big(-\frac{\big(l-\tau_{s',s}(t)\big)^2}{\sigma_{\tau}}\Big)
$$

## 4.2.4 变系数时空自回归方程（完整形式）
$$
\begin{aligned}
\hat{y}_{s,t}= &\sum_{l\in[-L,L],\,l\neq0} A_{s,l}(t)\,\tilde{y}^{*}_{s,t+l} \\
&+\sum_{l\in[-L,L],\,l\neq0}\beta^{\text{sp}}_{s,l}(t)\,
\Big[\sum_{s'\neq s}^{S}\Omega\big(l,\tau_{s',s}(t)\big)\,\tilde{\alpha}_{s',s}(t)\,\tilde{y}^{*}_{s',t+l}\Big] \\
&+\gamma_{s}(t)\,
\Big[\sum_{s'\neq s}^{S}\Omega\big(0,\tau_{s',s}(t)\big)\,\tilde{\alpha}_{s',s}(t)\,\tilde{y}^{*}_{s',t}\Big] \\
&+\sum_{p=1}^{P} B_{s,p}(t)\,x^{(p)}_{s,t}.
\end{aligned}
$$
其中 $\tilde{y}^{*}_{s,t}=r_{s,t}y_{s,t}+(1-r_{s,t})y^{\text{clean}}_{s,t}$ 为观测/填充混合。

## 4.2.5 Transformer 超网络权参分发机制
1. **输入嵌入**：将 $\mathbf{Y},\mathbf{R},\Theta,\Lambda,Z,\mathbf{X}$ 拼接后经线性投影得到 $\mathbf{E}\in\mathbb{R}^{S\times T\times d_{model}}$。
2. **时序 Self‑Attention Encoder**：捕获跨时间的长程依赖。
3. **空间 Self‑Attention Encoder**：捕获跨站点的空间关联（包括 $\tilde{\alpha}$ 与 $\Omega$ 信息）。
4. **解耦投影头**：四个独立的全连接层分别输出 $A,\beta^{\text{sp}},\gamma,B$，以及清洗基底 $\mathbf{y}^{\text{clean}}$。

---
