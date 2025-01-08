import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import gridspec

###############################################################################
# 1) Define color maps
###############################################################################
def generate_far_colormap():
    """FAR colormap: 0–20% = green, up to 100% = red."""
    colors = [
        "#008000",  # 0–20
        "#ADFF2F",  # 20–40
        "#FFFF00",  # 40–60
        "#FFA500",  # 60–80
        "#FF0000",  # 80–100
    ]
    boundaries = [0,20,40,60,80,100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return cmap, norm

def generate_hr_colormap():
    """HR colormap: 0–20% = red, up to 100% = green."""
    colors = [
        "#FF0000",  # 0–20
        "#FFA500",  # 20–40
        "#FFFF00",  # 40–60
        "#ADFF2F",  # 60–80
        "#008000",  # 80–100
    ]
    boundaries = [0,20,40,60,80,100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return cmap, norm

###############################################################################
# 2) Example data: 4 rows × 6 columns
#    rows => [Wajir MAM, Marsabit MAM, Wajir OND, Marsabit OND]
#    cols => [Mar, Apr, May, Oct, Nov, Dec]
###############################################################################
# FAR data
far_moderate = np.full((4,6), np.nan)
far_moderate[0,0], far_moderate[0,1], far_moderate[0,2] = 43.4, 40.4, 46.5
far_moderate[1,0], far_moderate[1,1], far_moderate[1,2] = 47.5, 39.4, 51.5
far_moderate[2,3], far_moderate[2,4], far_moderate[2,5] = 43.4, 40.4, 40.4
far_moderate[3,3], far_moderate[3,4], far_moderate[3,5] = 39.4, 38.4, 43.4

far_extreme = np.full((4,6), np.nan)
far_extreme[0,0], far_extreme[0,1], far_extreme[0,2] = 18.2, 15.2, 17.2
far_extreme[1,0], far_extreme[1,1], far_extreme[1,2] = 24.2, 15.2, 20.2
far_extreme[2,3], far_extreme[2,4], far_extreme[2,5] = 20.2, 18.2, 20.2
far_extreme[3,3], far_extreme[3,4], far_extreme[3,5] = 13.1, 11.1, 23.2

far_severe = np.full((4,6), np.nan)
far_severe[0,0], far_severe[0,1], far_severe[0,2] = 39.4, 33.3, 41.4
far_severe[1,0], far_severe[1,1], far_severe[1,2] = 35.4, 33.3, 39.4
far_severe[2,3], far_severe[2,4], far_severe[2,5] = 31.3, 23.2, 28.3
far_severe[3,3], far_severe[3,4], far_severe[3,5] = 17.2, 22.2, 33.3

# HR data
hr_moderate = far_moderate.copy()
hr_extreme  = far_extreme.copy()
hr_severe   = far_severe.copy()

###############################################################################
# 3) Region & SPI labels
###############################################################################
region_labels = ["Wajir", "Marsabit", "Wajir", "Marsabit"]
spi_labels    = ["MAM",   "MAM",      "OND",   "OND"]

###############################################################################
# 4) Generic function to plot any 3 categories (Moderate, Extreme, Severe),
#    with 2 text columns on the left (Region, SPI), a color map, 
#    increased figure height, and a bottom title placed well below the color bar.
###############################################################################
def plot_three_blocks_with_text(
    data_mod, data_ext, data_sev,
    region_lbls, spi_lbls,
    cmap, norm,
    metric_name="False Alarm Ratio (%)",
    output_file="FAR_no_overlap.png"
):
    # Increase the figure height, e.g. 8
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(nrows=1, ncols=5, width_ratios=[1.2,1.2,5,5,5], wspace=0.6)

    ax_region = fig.add_subplot(gs[0])
    ax_spi    = fig.add_subplot(gs[1])
    ax_mod    = fig.add_subplot(gs[2])
    ax_ext    = fig.add_subplot(gs[3])
    ax_sev    = fig.add_subplot(gs[4])

    ax_region.axis("off")
    ax_spi.axis("off")

    # We'll position 4 row labels, spaced vertically
    y_positions = [0.8, 0.55, 0.3, 0.05]
    for (reg,y) in zip(region_lbls, y_positions):
        ax_region.text(0.5, y, reg, ha="center", va="center", fontsize=11)

    for (spi,y) in zip(spi_lbls, y_positions):
        ax_spi.text(0.5, y, spi, ha="center", va="center", fontsize=11)

    # columns => [Mar, Apr, May, Oct, Nov, Dec]
    col_labels = ["Mar","Apr","May","Oct","Nov","Dec"]
    x_ticks = np.arange(6)+0.5
    y_ticks = np.arange(4)+0.5

    annot_kws = dict(fontsize=10, ha="center", va="center")

    #--- moderate
    sns.heatmap(
        data_mod,
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        linewidths=1, linecolor="white",
        cbar=False,
        square=False,
        ax=ax_mod
    )
    ax_mod.set_title("Moderate", fontsize=12, pad=5)
    ax_mod.set_xticks(x_ticks)
    ax_mod.set_xticklabels(col_labels, rotation=0)
    ax_mod.xaxis.set_ticks_position("top")
    ax_mod.xaxis.set_label_position("top")
    ax_mod.set_yticks(y_ticks)
    ax_mod.set_yticklabels([""]*4)
    ax_mod.set_ylim(4,0)

    #--- extreme
    sns.heatmap(
        data_ext,
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        linewidths=1, linecolor="white",
        cbar=False,
        square=False,
        ax=ax_ext
    )
    ax_ext.set_title("Extreme", fontsize=12, pad=5)
    ax_ext.set_xticks(x_ticks)
    ax_ext.set_xticklabels(col_labels, rotation=0)
    ax_ext.xaxis.set_ticks_position("top")
    ax_ext.xaxis.set_label_position("top")
    ax_ext.set_yticks(y_ticks)
    ax_ext.set_yticklabels([""]*4)
    ax_ext.set_ylim(4,0)

    #--- severe
    sns.heatmap(
        data_sev,
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        linewidths=1, linecolor="white",
        cbar=False,
        square=False,
        ax=ax_sev
    )
    ax_sev.set_title("Severe", fontsize=12, pad=5)
    ax_sev.set_xticks(x_ticks)
    ax_sev.set_xticklabels(col_labels, rotation=0)
    ax_sev.xaxis.set_ticks_position("top")
    ax_sev.xaxis.set_label_position("top")
    ax_sev.set_yticks(y_ticks)
    ax_sev.set_yticklabels([""]*4)
    ax_sev.set_ylim(4,0)

    # layout: enough bottom space for color bar + final title
    plt.tight_layout(rect=[0,0.25,1,0.98])

    # color bar: narrower width, placed around y=0.14
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.4, 0.16, 0.2, 0.03])  # left,bottom,width,height
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0,20,40,60,80,100])
    # e.g.: <20, 20-40, 40-60, 60-80, 80<, 100
    if "Hit Rate" in metric_name:
        cbar.set_ticklabels(["<20","20-40","40-60","60-80","80-100","100"])
    else:
        cbar.set_ticklabels(["<20","20-40","40-60","60-80","80<","100"])
    cbar.set_label(metric_name, fontsize=10)

    # main title even lower => y=0.10, well below the color bar
    fig.text(
        0.5, 0.10,
        f"{metric_name} - Wajir & Marsabit (MAM & OND)",
        ha="center", va="center", fontsize=12
    )

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


###############################################################################
# 5) Generate both the FAR and HR final figures with no overlap
###############################################################################
if __name__ == "__main__":
    # FAR
    far_cmap, far_norm = generate_far_colormap()
    plot_three_blocks_with_text(
        far_moderate, far_extreme, far_severe,
        region_labels, spi_labels,
        far_cmap, far_norm,
        metric_name="False Alarm Ratio (%)",
        output_file="FAR_no_overlap.png"
    )

    # HR
    hr_cmap, hr_norm = generate_hr_colormap()
    plot_three_blocks_with_text(
        hr_moderate, hr_extreme, hr_severe,
        region_labels, spi_labels,
        hr_cmap, hr_norm,
        metric_name="Hit Rate (%)",
        output_file="HR_no_overlap.png"
    )

    print("Saved FAR_no_overlap.png and HR_no_overlap.png")
