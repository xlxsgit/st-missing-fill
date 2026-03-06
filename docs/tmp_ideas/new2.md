# 基于多源地形与气象特征的风速插补定制算法升级

在上一版推导（`new.md`）中，我们建立了基于风流体平流时间对齐的元素级自回归方程。现在，随着极其丰富的**高精度位置坐标、微观立体地形图谱以及多元气象协变量**介入，我们可以将该算法从“理想平原风洞实验”彻底升格为具有极强现实物理意义的**“阿尔卑斯高山-峡谷阻滞与导流物理模型”**。

---

## 1. 基础坐标体系精算（摆脱球面经纬度误差）

表 2 提供了 **LV95 瑞士投影坐标系 (X, Y)**，这是一种米级的笛卡尔正交投影系统，极具价值！我们直接用它替代球面 `LAT, LON`，可以无损且极速地算出物理底座：

**目标站点 $s$ 与 源站点 $s'$ 之间的精确传输距离 $D_{s',s}$:**
$$ D_{s',s} = \sqrt{(X_s - X_{s'})^2 + (Y_s - Y_{s'})^2} $$

**精确的绝对来风方位角 $\Phi_{s',s}$ (基准为北=0°, 东=90°):**
$$ \Phi_{s',s} = \left( 90^\circ - \frac{180^\circ}{\pi} \cdot \text{atan2}(Y_s - Y_{s'}, \ X_s - X_{s'}) \right) \mod 360^\circ $$
*(通过 $\Phi$ 我们立刻就能以最高精度计算出上版文档中的对风夹角 $\Delta \theta_{s',s}$)*

---

## 2. 核心大更新：三维地形阻力与阻滞效应 (Topographic Friction)

上一版本中，空间传播权重 $\alpha$ 只有二维的风向衰减。但现实中，在多山的瑞士，风的传播经常被山脉横切断或顺延峡谷加速。表 3 提供的变量 `{Z (高程), ASPECT (坡向), SLOPE (坡度), TPI (地形特征)}` 完美弥补了这一空缺。

我们将原有的二维权重 $\alpha$ 升级为**三维地形感知传播权重 $\tilde{\alpha}_{s',s}(t-l)$**：

### 2.1 高级垂直衰变惩罚 (Elevation Gradient Penalty)
空气从 $s'$ 流向 $s$ 需要跨越山陵。如果高程 $Z$ 差距极大（比如从海拔2000米吹向海拔500米），热力学和机械湍流会急剧损耗能量。
我们增加一个针对相对高差的垂直指数惩罚项： 
$$ \text{Penalty}_{Z} = \exp\left( - \frac{|Z_s - Z_{s'}|}{\Sigma_z} \right) $$

### 2.2 坡向迎风截面阻力（Micro-scale Aspect Blocking）
当风带着特定的方向吹到站点 $s$ 附近时，若 $s$ 的 **ASPECT (2KM范围坡向)** 与吹来的风向相反（即站点在背风坡），实际测得的风速将面临剧烈的地形削弱摩擦。
定义目标站的地形“迎风吻合增益” $\mu_s(t-l)$：
$$ \mu_{s}(t-l) = \text{ReLU}\Big(\cos\big( \Theta_{s', t-l} - \text{ASPECT}_s \big)\Big) $$
（如果完全顺风迎坡而上 $\mu=1$，如果在山后的背风坡 $\mu=0$ 产生涡流效应，直接断绝远距离平流影响）。

### 升级后的地形权重公式汇总：
$$
\tilde{\alpha}_{s',s}(t-l) = 
\begin{cases} 
\cos\big( \Delta \theta_{s',s} \big) \cdot \exp\left( - \frac{D_{s',s}}{\Sigma_d} - \frac{|Z_s - Z_{s'}|}{\Sigma_z} \right) \cdot \big(1 + \lambda_{topo} \cdot \mu_s(t-l)\big), & \Delta \theta \le 90^\circ \\
0, & \Delta \theta > 90^\circ
\end{cases}
$$
*常数 $\Sigma_d, \Sigma_z, \lambda_{topo}$ 可以在网络中设为可学习的 Parameter。*

---

## 3. 基于 10 分钟平均风速的传导时延 $\tau$ 设定

表 4 给出了极为关键的细节：数据时间精度为 **10分钟/步**，且提供了**平均风速**。
由于我们的时间序列模型以 10 分钟为一个固定步长，流体在空间中平流输送的速度也完全交由该时段内的代表性标量（10分钟平均风速）定义。我们以最纯粹的方法计算这股风从上游输送到下游目标的耗时：

计算时，使用源站点前一轮插补出的（或者原始观测的）有效平均风速 $V^{\text{eff}}_{s',s} = \text{MeanWind}_{s'} \cdot \cos(\Delta \theta_{s',s})$。

更新后的预期耗时计算式（统一转化为以 10 分钟为 1 个模型步长 $\text{steps}$）：
$$ \tau_{s',s}(t-l) = \left( \frac{D_{s',s}}{ \hat{Y}_{s',t-l}^{(I)} \cdot \cos(\Delta \theta_{s',s}(t-l)) + \epsilon} \right) \div 600\text{秒} $$

---

## 4. 全局模型与表征方程的组装 (The Grand Architecture)

有了这么详尽的外部动态变量 $X$，我们需要把它们按其**物理职责**塞进先前的因果时间循环等式中。这里有两类变量：
1. **直接驱动因素（Pressure 气压等）**：气压差本身就是产生风的原因；高点与低点的气压梯度主导了局部风的新生。
2. **大气层结稳定度因素（Temp 温度，湿度，降水，日照等）**：它们决定了空气是对流旺盛（更易将高空强风传导致地面测站）还是逆温锁死（各刮各的，空间弱相关）。

最终，带有瑞士地理特色及完全物理定标的风速自回归迭代预测模型为：

$$
\begin{align*}
\hat{Y}_{s,t}^{(I+1)} = \ &\sum_{l=1}^L A_l(t) \cdot \hat{Y}_{s,t-l}^{(I)} \quad \text{[本地序列的时间惯性守恒]} \\ 
&+ \sum_{l=1}^L \beta_l \cdot \left[ \sum_{s' \neq s}^S \underbrace{ \Omega\Big(l, \tau_{s',s}(t-l)\Big) }_{\text{到达时间流核}} \cdot \underbrace{ \tilde{\alpha}_{s',s}(t-l) }_{\text{3D地形感知传播衰减}} \cdot \underbrace{ \hat{Y}_{s',t-l}^{(I)} }_{\text{上风向风速母体}} \right] \\
&+ F_{MLP} \Big( \mathbf{X}_{s,t}^{\text{Meteo}}, \ \mathbf{X}_{s}^{\text{TopoStatic}} \Big) \quad \text{[多模态局部微气候修正项]}\\
&+ \epsilon_{s,t}
\end{align*}
$$

**核心变化总结**：
后侧协变量项不再是简单的线性累加 $\sum B_p X^P$，而是通过一个微型神经网络 $F_{MLP}$ 去融合。
输入包含静态节点属性：$\{\text{SLOPE, STD, TPI, D\_WE, D\_SN}\}$，
以及动态节点气象：$\{\text{相对湿度, 降水, 气压, 日照, 气温}\}$。
这类融合允许模型学习到类似于 _“当 TPI<0 (身处峡谷) 且 日照=0 且有强降水下击暴流时，产生难以通过周边大尺度上游来推断的剧烈局部怪风”_ 的非线性异常模式。

这样，**大空间尺度的因果传递**交给了前方的 $\Omega$ 时空对齐物理白盒；而**站点的非线性局地微气象**交给了后面的 $F_{MLP}$ 潜特征黑盒，结构极其优雅，可解释性极高。
