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
import os
from dotenv import load_dotenv

import dask.dataframe as dd
import geopandas as gpd
from shapely import wkb
import regionmask

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from google.oauth2 import service_account

def regionmask_three_polygons():
    """
    1) Load environment variables (.env),
    2) Access GCS parquet + credentials,
    3) Filter for three region IDs:
       - 45 => gambela (ETH-ADM1-3_0_0-B6)
       - 82 => marsabit (KEN-ADM1-3_0_0-B32)
       - 128 => jonglei (SSD-ADM1-3_0_0-B8)
    4) Combine polygons into one regionmask,
    5) Create synthetic data, mask, and plot all regions.
    """

    # 1) Load environment variables from your .env
    load_dotenv("/Users/sinugp/Downloads/env")

    parquet_url = os.getenv("REGION_PARQUET_PATH")
    service_account_json = os.getenv("CREDENTIALS_PATH")

    # We have region_filter_map for each region ID
    region_filter_map = {
        45: "ETH-ADM1-3_0_0-B6",    # Gambela
        82: "KEN-ADM1-3_0_0-B32",   # Marsabit
        128: "SSD-ADM1-3_0_0-B8",   # Jonglei
    }
    region_name_dict = {
        45: "Gambela",
        82: "Marsabit",
        128: "Jonglei",
    }

    # 2) Create GCS credentials
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
    )

    # Read the entire parquet with Dask
    print(f"Reading parquet from {parquet_url}")
    ddf = dd.read_parquet(
        parquet_url,
        engine="pyarrow",
        storage_options={"token": credentials},
    )
    # We'll collect polygons for our 3 region IDs
    outlines = []
    names = []

    # 3) For each region_id => filter df, get geometry
    for reg_id, reg_filter in region_filter_map.items():
        fdf = ddf[ddf["gbid"].str.contains(reg_filter, case=False, na=False)]
        df_local = fdf.compute()
        if df_local.empty:
            raise ValueError(
                f"No rows matched region_filter='{reg_filter}' in {parquet_url}"
            )
        # Convert WKB → geometry
        df_local["geometry"] = df_local["geometry"].apply(wkb.loads)
        gdf_local = gpd.GeoDataFrame(df_local, geometry="geometry")

        # In typical usage, you'd see just 1 row for each region
        # or possibly multiple. Let's pick the first geometry if multiple:
        geom = gdf_local.iloc[0].geometry

        outlines.append(geom)
        names.append(region_name_dict[reg_id])

    # 4) Build regionmask with these 3 polygons
    # We set overlap=False so regionmask can produce a single 2D mask
    # even if edges overlap. If there's actual area overlap, regionmask will
    # either produce warnings or require a 3D mask.
    three_regions = regionmask.Regions(
        outlines=outlines,
        names=names,
        name="EA_three",
        overlap=False
    )
    print("Built a regionmask with 3 polygons: ", three_regions)

    # 5) Create synthetic data, say lat=0..10, lon=28..40, so it covers all
    #    Adjust if needed for your actual polygon extents
    lat = np.linspace(0, 10, 51)
    lon = np.linspace(28, 40, 121)
    data = np.random.rand(len(lat), len(lon))

    ds = xr.Dataset(
        {"mydata": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon}
    )

    # regionmask <0.9 => pass ds["lon"], ds["lat"]
    # shape => (lat, lon), each cell = 0,1,2 or NaN outside polygons
    mask_2d = three_regions.mask(ds["lon"], ds["lat"])

    # For data in *all* polygons, we can keep mask_2d >= 0 (i.e. any region).
    ds_masked = ds.where(mask_2d >= 0)

    # 6) Plot
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    # Plot data from all polygons => masked outside
    ds_masked["mydata"].plot(
        ax=ax,
        x="lon",
        y="lat",
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "Random Data"}
    )

    # Draw polygons
    # 'add_label=True' draws each polygon label, but usually we set 'False'
    three_regions.plot(ax=ax, line_kws=dict(color="red", linewidth=2), add_label=True)

    # Zoom to lat=0..10, lon=28..40
    ax.set_extent([28, 40, 0, 10], crs=ccrs.PlateCarree())

    ax.set_title("Regionmask: Marsabit, Gambela, Jonglei from GCS Parquet")
    ax.coastlines()
    plt.savefig("ea_3regions_marsabit_gambela_jonglei.png", dpi=100, bbox_inches="tight")
    plt.show()

    print("\nSaved 'ea_3regions_marsabit_gambela_jonglei.png'. Done.")

if __name__ == "__main__":
    regionmask_three_polygons()

