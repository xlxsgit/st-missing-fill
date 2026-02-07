import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.data.load import load_data
from src.data.misser import mcar_masker, seq_masker, scm_masker, mask_mapping

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

def plot_miss_im(masked_list):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 7),sharex=True)

    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.08, top=0.96, hspace=0.12)

    for ax, masked_y, title in zip(axes, masked_list, ["MCAR", "SEQ", "SCM"]):
        im = ax.imshow(masked_y[:, 1000:2024],
                       aspect="auto", origin="lower", 
                       cmap='viridis')
        ax.set_title(title, loc="left", fontweight="bold")

    axes[-1].set_xlabel("Time index")
    ylab = fig.supylabel("Station (sorted by cluster)", x=0.045)  # 微调 x 位置配合 left 边距

    cbar = fig.colorbar( # 【关键】调整 colorbar 宽度和间距
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.018,   # 缩窄 colorbar
        pad=0.015         # 减小与主图间距
    )
    cbar.set_label("Observed value")

    plt.savefig('data/figs/single_special_plot/missing_pattern.png', dpi=400, bbox_inches='tight')
    plt.close()


def plot_missing_pattern():
    ground_X, ground_y, all_stations, all_stations_cluster, vars = load_data()

    mask_mcar = mcar_masker(shape=ground_y.shape, pi = 0.3)
    mask_seq = seq_masker(shape=ground_y.shape, pi=0.3)
    mask_scm = scm_masker(S_cluster=all_stations_cluster, shape=ground_y.shape, pi=0.3)

    y_mcar = mask_mapping(mask_mcar, ground_y)
    y_seq = mask_mapping(mask_seq, ground_y)
    y_scm = mask_mapping(mask_scm, ground_y)
    
    plot_miss_im(masked_list=[y_mcar, y_seq, y_scm])

if __name__=='__main__':
    print("Plotting missing pattern...")
    plot_missing_pattern()