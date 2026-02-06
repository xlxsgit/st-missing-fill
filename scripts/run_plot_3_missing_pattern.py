import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.data.misser import mask_mcar, mask_seq, mask_scm

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


data = pd.read_parquet('data/processed/all_data.parquet')
data_cols = data.columns.tolist()
stations = sorted(pd.read_csv('data/processed/all_stations.csv', usecols=['Station'])['Station'].tolist())
vars = pd.read_csv('data/raw/geo_data/vars.csv', usecols=['name'])['name'].tolist()
vars = {'y': 'wind_speed', 'x': [var for var in vars if var != 'wind_speed']}
S_cluster = pd.read_csv('data/processed/all_stations.csv')['cluster'].tolist()


ground_y = data[[f'{s}~{vars["y"]}' for s in stations]].to_numpy().transpose()

mask1 = mask_mcar(size=ground_y.shape)
mask2 = mask_seq(size=ground_y.shape)
mask3 = mask_scm(S_cluster, size=ground_y.shape)

masked_y1 = np.where(mask1 == 1, ground_y, np.nan)
masked_y2 = np.where(mask2 == 1, ground_y, np.nan)
masked_y3 = np.where(mask3 == 1, ground_y, np.nan)

order = np.argsort(S_cluster)
masked_y1 = masked_y1[order]
masked_y2 = masked_y2[order]
masked_y3 = masked_y3[order]

# =========================
# 画图参数（保持不变）
# =========================
a, delta_a = 1000, 1024
masked_list = [masked_y1, masked_y2, masked_y3]
titles = ["MCAR", "SEQ", "SCM"]

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgrey")
vmin = np.nanpercentile(ground_y, 5)
vmax = np.nanpercentile(ground_y, 95)

# =========================
# 创建子图 + 手动布局调整（关键修改）
# =========================
fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(14, 7),
    sharex=True
)

# 【关键】手动调整边距和子图间距（替代 tight_layout）
fig.subplots_adjust(
    left=0.09,    # 左边距：为 y 轴标签留空间
    right=0.93,   # 右边距：为 colorbar 留空间
    bottom=0.08,  # 底边距：为 x 轴标签留空间
    top=0.96,     # 顶边距：为标题留空间
    hspace=0.12   # 子图垂直间距（减小空白）
)

for ax, masked_y, title in zip(axes, masked_list, titles):
    im = ax.imshow(
        masked_y[:, a:a + delta_a],
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none"
    )
    ax.set_title(title, loc="left", fontweight="bold")

axes[-1].set_xlabel("Time index")
ylab = fig.supylabel("Station (sorted by cluster)", x=0.045)  # 微调 x 位置配合 left 边距

# 【关键】调整 colorbar 宽度和间距
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.018,   # 缩窄 colorbar
    pad=0.015         # 减小与主图间距
)
cbar.set_label("Observed value")

plt.savefig('data/figs/single_special_plot/missing_pattern.png', dpi=400, bbox_inches='tight')
plt.show()