# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (base)
#     language: python
#     name: base
# ---

# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Mock function to simulate the colormap and normalization creation
def generate_custom_colormap(reverse_colors=False):
    """
    Mock function to generate a custom colormap and normalization for testing.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF"]  # Example colors
    if reverse_colors:
        colors = colors[::-1]

    boundaries = [0, 20, 40, 60, 80, 100]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    return cmap, norm

# Mock data for dt_df
dt_df = {
    "mod": {
        "annot_hr": np.random.randint(0, 100, (5, 3)),
        "annot_far": np.random.randint(0, 100, (5, 3)),
        "data": np.random.randint(0, 100, (5, 3)),
    },
    "sev": {
        "annot_hr": np.random.randint(0, 100, (5, 3)),
        "annot_far": np.random.randint(0, 100, (5, 3)),
        "data": np.random.randint(0, 100, (5, 3)),
    },
    "ext": {
        "annot_hr": np.random.randint(0, 100, (5, 3)),
        "annot_far": np.random.randint(0, 100, (5, 3)),
        "data": np.random.randint(0, 100, (5, 3)),
    },
}

# Mock params object
class MockParams:
    def __init__(self):
        self.output_path = "./"
        self.region_id = "test_region"
        self.sc_season_str = "test_season"

params = MockParams()

# Import and test the create_heatmap_subplot function
def test_create_heatmap_subplot():
    """
    Test the create_heatmap_subplot function with mock data and parameters.
    """
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 18))
    categories = ["mod", "sev", "ext"]
    titles = ["Mild", "Moderate", "Severe"]

    hr_cmap, hr_norm = generate_custom_colormap(reverse_colors=True)
    far_cmap, far_norm = generate_custom_colormap(reverse_colors=False)

    # Plot Hit Rates (HR)
    for i, (cat, title) in enumerate(zip(categories, titles)):
        sns.heatmap(
            dt_df[cat]["annot_hr"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=hr_cmap,
            norm=hr_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[0, i],
            square=True,
        )
        axes[0, i].set_title(f"{title} - HR Row 1")

        sns.heatmap(
            dt_df[cat]["annot_hr"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=hr_cmap,
            norm=hr_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[1, i],
            square=True,
        )
        axes[1, i].set_title(f"{title} - HR Row 2")

    # Plot False Alarm Ratios (FAR)
    for i, (cat, title) in enumerate(zip(categories, titles)):
        sns.heatmap(
            dt_df[cat]["annot_far"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=far_cmap,
            norm=far_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[2, i],
            square=True,
        )
        axes[2, i].set_title(f"{title} - FAR Row 1")

        sns.heatmap(
            dt_df[cat]["annot_far"],
            annot=dt_df[cat]["data"],
            fmt=".1f",
            cmap=far_cmap,
            norm=far_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
            ax=axes[3, i],
            square=True,
        )
        axes[3, i].set_title(f"{title} - FAR Row 2")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Hit Rates and False Alarm Ratios by Category", fontsize=16, y=0.98)

    # Add colorbars
    cbar_ax_hr = fig.add_axes([1.02, 0.55, 0.02, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(cmap=hr_cmap, norm=hr_norm), cax=cbar_ax_hr)
    cbar_ax_hr.set_title("HR")

    cbar_ax_far = fig.add_axes([1.02, 0.15, 0.02, 0.35])
    plt.colorbar(plt.cm.ScalarMappable(cmap=far_cmap, norm=far_norm), cax=cbar_ax_far)
    cbar_ax_far.set_title("FAR")

    # Save figure
    output_file = f"{params.output_path}dt_{params.region_id}_{params.sc_season_str}_test.png"
    print(f"Saving test heatmap to: {output_file}")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Run the test
test_create_heatmap_subplot()

# -




