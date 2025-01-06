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

###############################################################################
# 1) Custom colormap function (same as your mock version)
###############################################################################
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

###############################################################################
# 2) Replace the dummy dataframe with actual values for Marsabit & Wajir
#    (MAM & OND). We'll store 4 rows x 3 columns:
#       Row 0 = Marsabit MAM
#       Row 1 = Wajir   MAM
#       Row 2 = Marsabit OND
#       Row 3 = Wajir   OND
#    Columns correspond to that season's 3 months.
###############################################################################
dt_df = {
    "mod": {
        # -------- Hit Rate (HR) --------
        "annot_hr": np.array([
            [47.5, 39.4, 51.5],  # Marsabit MAM (Feb, Jan, Dec)
            [43.4, 40.4, 46.5],  # Wajir   MAM (Feb, Jan, Dec)
            [39.4, 38.4, 43.4],  # Marsabit OND (Sep, Aug, Jul)
            [43.4, 40.4, 40.4],  # Wajir   OND (Sep, Aug, Jul)
        ]),
        # -------- False Alarm Ratio (FAR) --------
        "annot_far": np.array([
            [47.5, 39.4, 51.5],  # Marsabit MAM (Feb, Jan, Dec)
            [43.4, 40.4, 46.5],  # Wajir   MAM
            [39.4, 38.4, 43.4],  # Marsabit OND (Sep, Aug, Jul)
            [43.4, 40.4, 40.4],  # Wajir   OND
        ]),
        # The 'data' array is what Seaborn will use to annotate the cells.
        # We'll just copy the Hit Rate values for demonstration.
        "data": np.array([
            [47.5, 39.4, 51.5],
            [43.4, 40.4, 46.5],
            [39.4, 38.4, 43.4],
            [43.4, 40.4, 40.4],
        ]),
    },
    "sev": {
        "annot_hr": np.array([
            [35.4, 33.3, 39.4],  # Marsabit MAM
            [39.4, 33.3, 41.4],  # Wajir   MAM
            [17.2, 22.2, 33.3],  # Marsabit OND
            [31.3, 23.2, 28.3],  # Wajir   OND
        ]),
        "annot_far": np.array([
            [35.4, 33.3, 39.4],  # Marsabit MAM
            [39.4, 33.3, 41.4],  # Wajir   MAM
            [17.2, 22.2, 33.3],  # Marsabit OND
            [31.3, 23.2, 28.3],  # Wajir   OND
        ]),
        "data": np.array([
            [35.4, 33.3, 39.4],
            [39.4, 33.3, 41.4],
            [17.2, 22.2, 33.3],
            [31.3, 23.2, 28.3],
        ]),
    },
    "ext": {
        "annot_hr": np.array([
            [24.2, 15.2, 20.2],  # Marsabit MAM
            [18.2, 15.2, 17.2],  # Wajir   MAM
            [13.1, 11.1, 23.2],  # Marsabit OND
            [20.2, 18.2, 20.2],  # Wajir   OND
        ]),
        "annot_far": np.array([
            [24.2, 15.2, 20.2],  # Marsabit MAM
            [18.2, 15.2, 17.2],  # Wajir   MAM
            [13.1, 11.1, 23.2],  # Marsabit OND
            [20.2, 18.2, 20.2],  # Wajir   OND
        ]),
        "data": np.array([
            [24.2, 15.2, 20.2],
            [18.2, 15.2, 17.2],
            [13.1, 11.1, 23.2],
            [20.2, 18.2, 20.2],
        ]),
    },
}

###############################################################################
# 3) Mock params object (unchanged)
###############################################################################
class MockParams:
    def __init__(self):
        self.output_path = "./"
        self.region_id = "test_region"
        self.sc_season_str = "test_season"

params = MockParams()

###############################################################################
# 4) The same plotting test function, but now it will plot the actual MAM/OND
#    data for Wajir and Marsabit that we inserted in dt_df above.
###############################################################################
def test_create_heatmap_subplot():
    """
    Test the create_heatmap_subplot function with updated data 
    for Marsabit/Wajir (MAM & OND).
    """
    from matplotlib.colors import Normalize

    # 4 subplot-rows, 3 columns -> 12 subplots total.
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 18))

    # Our categories in dt_df are "mod", "sev", "ext".
    # We'll rename them in the figure as "Moderate", "Severe", "Extreme".
    categories = ["mod", "sev", "ext"]
    titles = ["Moderate", "Severe", "Extreme"]

    hr_cmap, hr_norm = generate_custom_colormap(reverse_colors=True)
    far_cmap, far_norm = generate_custom_colormap(reverse_colors=False)

    # Plot Hit Rates (HR) in rows 0 and 1
    for i, (cat, title) in enumerate(zip(categories, titles)):
        # First row of subplots (row=0) for HR
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

        # Second row of subplots (row=1) for HR
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

    # Plot False Alarm Ratios (FAR) in rows 2 and 3
    for i, (cat, title) in enumerate(zip(categories, titles)):
        # Third row of subplots (row=2)
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

        # Fourth row of subplots (row=3)
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
    fig.suptitle("Hit Rates and False Alarm Ratios (Wajir & Marsabit; MAM & OND)", 
                 fontsize=16, y=0.98)

    # Add colorbars
    cbar_ax_hr = fig.add_axes([1.02, 0.55, 0.02, 0.35])
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=hr_cmap, norm=hr_norm),
        cax=cbar_ax_hr
    ).set_label("HR (%)")

    cbar_ax_far = fig.add_axes([1.02, 0.15, 0.02, 0.35])
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=far_cmap, norm=far_norm),
        cax=cbar_ax_far
    ).set_label("FAR (%)")

    # Save figure
    output_file = f"{params.output_path}wajir_marsabit.png"
    print(f"Saving test heatmap to: {output_file}")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

###############################################################################
# 5) Run the test
###############################################################################
test_create_heatmap_subplot()

