import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

###############################################################################
# 1) Define colormaps for Hit Rate (red→green) and FAR (green→red)
###############################################################################
def generate_hr_colormap():
    """Hit Rate colormap: 0% = red, 100% = green, binned at [0,20,40,60,80,100]."""
    colors = [
        "#FF0000",  # 0–20
        "#FFA500",  # 20–40
        "#FFFF00",  # 40–60
        "#ADFF2F",  # 60–80
        "#008000",  # 80–100
    ]
    boundaries = [0, 20, 40, 60, 80, 100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return cmap, norm

def generate_far_colormap():
    """FAR colormap: 0% = green, 100% = red, binned at [0,20,40,60,80,100]."""
    colors = [
        "#008000",  # 0–20
        "#ADFF2F",  # 20–40
        "#FFFF00",  # 40–60
        "#FFA500",  # 60–80
        "#FF0000",  # 80–100
    ]
    boundaries = [0, 20, 40, 60, 80, 100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return cmap, norm

###############################################################################
# 2) Create sample data for HR or FAR in shape (4, 6):
#    rows => [Wajir MAM, Marsabit MAM, Wajir OND, Marsabit OND]
#    columns => [Mar, Apr, May, Oct, Nov, Dec]
#    Replace these with your real numeric values.
###############################################################################
# Example: Hit Rate (Moderate)
hr_moderate = np.full((4,6), np.nan)

# row0: Wajir MAM => Mar,Apr,May
hr_moderate[0,0], hr_moderate[0,1], hr_moderate[0,2] = 43.4, 40.4, 46.5
# row1: Marsabit MAM => Mar,Apr,May
hr_moderate[1,0], hr_moderate[1,1], hr_moderate[1,2] = 47.5, 39.4, 51.5
# row2: Wajir OND => Oct,Nov,Dec
hr_moderate[2,3], hr_moderate[2,4], hr_moderate[2,5] = 43.4, 40.4, 40.4
# row3: Marsabit OND => Oct,Nov,Dec
hr_moderate[3,3], hr_moderate[3,4], hr_moderate[3,5] = 39.4, 38.4, 43.4

# Hit Rate (Extreme)
hr_extreme = np.full((4,6), np.nan)
hr_extreme[0,0], hr_extreme[0,1], hr_extreme[0,2] = 18.2, 15.2, 17.2
hr_extreme[1,0], hr_extreme[1,1], hr_extreme[1,2] = 24.2, 15.2, 20.2
hr_extreme[2,3], hr_extreme[2,4], hr_extreme[2,5] = 20.2, 18.2, 20.2
hr_extreme[3,3], hr_extreme[3,4], hr_extreme[3,5] = 13.1, 11.1, 23.2

# Hit Rate (Severe)
hr_severe = np.full((4,6), np.nan)
hr_severe[0,0], hr_severe[0,1], hr_severe[0,2] = 39.4, 33.3, 41.4
hr_severe[1,0], hr_severe[1,1], hr_severe[1,2] = 35.4, 33.3, 39.4
hr_severe[2,3], hr_severe[2,4], hr_severe[2,5] = 31.3, 23.2, 28.3
hr_severe[3,3], hr_severe[3,4], hr_severe[3,5] = 17.2, 22.2, 33.3

# For FAR, you can do the same or reuse the same arrays for demonstration
far_moderate = hr_moderate.copy()
far_extreme = hr_extreme.copy()
far_severe  = hr_severe.copy()

###############################################################################
# 3) A function that plots three subplots (Moderate, Extreme, Severe) side-by-side,
#    each with 4x6 data, months across the top, row labels for region/spi on the left,
#    and a single colorbar at the bottom.
###############################################################################
def plot_three_categories(
    mod_data, ext_data, sev_data,
    metric_name="Hit Rate (%)",
    output_file="hit_rate_mam_ond.png"
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13,5), sharey=True)

    # Decide colormap: HR or FAR
    if "Hit Rate" in metric_name:
        cmap, norm = generate_hr_colormap()
    else:
        cmap, norm = generate_far_colormap()

    # Our 4 rows => Wajir MAM, Marsabit MAM, Wajir OND, Marsabit OND
    row_labels = ["Wajir MAM", "Marsabit MAM", "Wajir OND", "Marsabit OND"]
    # Our 6 columns => Mar, Apr, May, Oct, Nov, Dec
    col_labels = ["Mar","Apr","May","Oct","Nov","Dec"]

    # We'll center the tick labels in each cell
    x_positions = np.arange(len(col_labels)) + 0.5
    y_positions = np.arange(len(row_labels)) + 0.5

    # For text annotations
    annot_kws = dict(fontsize=10, ha="center", va="center")

    # Plot "Moderate" in axes[0]
    sns.heatmap(
        mod_data,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        mask=np.isnan(mod_data),  # hide blanks
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        linewidths=0, linecolor=None,
        square=False,  # let them be rectangular
        cbar=False,    # we'll add one colorbar for all
        ax=axes[0]
    )
    axes[0].set_title("Moderate", fontsize=12)
    # Row labels
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(row_labels, rotation=0)
    # Column labels on top
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(col_labels, rotation=0)
    axes[0].xaxis.set_ticks_position("top")
    axes[0].xaxis.set_label_position("top")

    # Plot "Extreme" in axes[1]
    sns.heatmap(
        ext_data,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        mask=np.isnan(ext_data),
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        linewidths=0, linecolor=None,
        square=False,
        cbar=False,
        ax=axes[1]
    )
    axes[1].set_title("Extreme", fontsize=12)
    # Hide y tick labels in the middle
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([""]*4)
    # Column labels on top
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(col_labels, rotation=0)
    axes[1].xaxis.set_ticks_position("top")
    axes[1].xaxis.set_label_position("top")

    # Plot "Severe" in axes[2]
    sns.heatmap(
        sev_data,
        annot=True, fmt=".1f", annot_kws=annot_kws,
        mask=np.isnan(sev_data),
        cmap=cmap, norm=norm,
        vmin=0, vmax=100,
        linewidths=0, linecolor=None,
        square=False,
        cbar=False,
        ax=axes[2]
    )
    axes[2].set_title("Severe", fontsize=12)
    # Hide y tick labels on the right
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels([""]*4)
    # Column labels
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(col_labels, rotation=0)
    axes[2].xaxis.set_ticks_position("top")
    axes[2].xaxis.set_label_position("top")

    # Adjust layout, leaving space for colorbar at the bottom
    plt.tight_layout(rect=[0,0.13,1,0.95])

    # Add one horizontal colorbar
    from matplotlib import cm
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])  # left, bottom, width, height
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(metric_name, fontsize=11)
    # Label each bin
    cbar.set_ticks([0,20,40,60,80,100])
    cbar.set_ticklabels(["<20","20-40","40-60","60-80","80-100","100"])

    fig.suptitle(f"{metric_name} - Wajir & Marsabit (MAM & OND)", fontsize=14, y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

###############################################################################
# 4) Finally, produce both HR and FAR figures
###############################################################################
if __name__ == "__main__":
    # Example: HR figure
    plot_three_categories(
        hr_moderate, hr_extreme, hr_severe,
        metric_name="Hit Rate (%)",
        output_file="HitRate_MAM_OND.png"
    )

    # Example: FAR figure
    plot_three_categories(
        far_moderate, far_extreme, far_severe,
        metric_name="False Alarm Ratio (%)",
        output_file="FAR_MAM_OND.png"
    )

    print("Saved HitRate_MAM_OND.png and FAR_MAM_OND.png")

