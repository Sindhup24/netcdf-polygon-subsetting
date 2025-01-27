#!/usr/bin/env python

"""
test_regionmask_seas5_chrips_multi.py

We have 5 region-groups with multiple polygons each:
  gmj => gambela, marsabit, jonglei
  wm  => wajir, makamba
  da  => djibouti, anseba region
  kb  => kigali city, bakool
  m   => mara

For each group:
 - We union the polygons from Parquet (no buffering => exact "map-like" shape).
 - We open the group’s CHIRPS netCDF, pick MAM => time=0 => shapely.vectorized.contains => mask
 - We open the group’s SEA51 netCDF, pick MAM => time=0 => same approach
 - We plot them side by side => row of subplots

Finally, we produce a single 5×2 figure => test_regionmask_seas5_chrips_multi.png
"""

import os
import dask.dataframe as dd
import geopandas as gpd
import shapely
import shapely.wkb
import shapely.vectorized
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from shapely.ops import unary_union
from dotenv import load_dotenv
from google.oauth2 import service_account

def main():
    load_dotenv("/Users/sinugp/Downloads/env")  # Adjust if needed

    # GCS Parquet path & credentials
    parquet_url = os.getenv("REGION_PARQUET_PATH")
    service_account_json = os.getenv("CREDENTIALS_PATH")
    if not parquet_url or not service_account_json:
        raise ValueError("Missing REGION_PARQUET_PATH or CREDENTIALS_PATH in .env")

    # Create credentials
    creds = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )

    print(f"Reading polygons from {parquet_url} ...")
    ddf = dd.read_parquet(
        parquet_url,
        engine="pyarrow",
        storage_options={"token": creds}
    )
    df_all = ddf.compute()
    # Convert WKB => shapely
    df_all["geometry"] = df_all["geometry"].apply(shapely.wkb.loads)
    gdf_all = gpd.GeoDataFrame(df_all, geometry="geometry")
    print(f"Full GDF => {len(gdf_all)} rows in the Parquet.\n")

    # 5 region-groups: (short_label, list_of_names, chirps_nc, seas51_nc)
    groups_info = [
        ("gmj", ["gambela","marsabit","jonglei"],       "gmj-25km-chirps-v2.0.monthly.nc", "gmj_fct_spi3.nc"),
        ("wm",  ["wajir","makamba"],                   "wm-25km-chirps-v2.0.monthly.nc",  "wm_fct_spi3.nc"),
        ("da",  ["djibouti","anseba region"],          "da-25km-chirps-v2.0.monthly.nc",  "da_fct_spi3.nc"),
        ("kb",  ["kigali city","bakool"],              "kb-25km-chirps-v2.0.monthly.nc",  "kb_fct_spi3.nc"),
        ("m",   ["mara"],                              "m-25km-chirps-v2.0.monthly.nc",   "m_fct_spi3.nc"),
    ]

    nrows = len(groups_info)
    ncols = 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4*nrows),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    if nrows == 1:
        axes = [axes]  # make it a list

    for i, (short_label, name_list, chirps_file, seas51_file) in enumerate(groups_info):
        row_axes = axes[i] if nrows>1 else axes[0]
        ax_ch, ax_fc = row_axes

        print(f"=== Group {i}: {short_label} => polygons={name_list}")
        # Filter polygons from gdf_all => union
        sub_gdf = gdf_all[(gdf_all["name"].isin(name_list)) & (gdf_all["level"]=="admin1")]
        if sub_gdf.empty:
            print(f"No polygons matched {name_list} => skipping row.")
            ax_ch.set_title(f"{short_label.upper()} => no polygons in parquet.")
            ax_fc.set_title(f"{short_label.upper()} => no polygons in parquet.")
            continue
        union_geom = unary_union(sub_gdf.geometry)
        if union_geom.is_empty:
            print(f"Union is empty => skip row.")
            ax_ch.set_title(f"{short_label.upper()} => empty union.")
            ax_fc.set_title(f"{short_label.upper()} => empty union.")
            continue

        # We'll keep this for bounding box + plotting
        minx, miny, maxx, maxy = union_geom.bounds
        gdf_union = gpd.GeoDataFrame(geometry=[union_geom], crs=sub_gdf.crs)

        # 1) CHIRPS => open => MAM => time=0 => shapely mask
        ds_ch_masked = None
        if not os.path.exists(chirps_file):
            print(f"CHIRPS file {chirps_file} not found => skip CHIRPS side for {short_label}.")
        else:
            ds_ch = xr.open_dataset(chirps_file)
            var_ch = "spi3" if "spi3" in ds_ch.data_vars else "precip"
            ds_ch_mam = ds_ch[var_ch].sel(time=ds_ch.time.dt.month.isin([3,4,5])).dropna(dim="time", how="all")
            if ds_ch_mam.time.size == 0:
                print("No MAM data => skip CHIRPS.")
            else:
                ds_ch_mam = ds_ch_mam.isel(time=0)
                lat_1d = ds_ch_mam["lat"].values
                lon_1d = ds_ch_mam["lon"].values
                lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
                mask_ch_2d = shapely.vectorized.contains(union_geom, lon2d, lat2d)
                ds_ch_masked = ds_ch_mam.where(mask_ch_2d)

        # 2) SEA51 => open => rename init->time => MAM => time=0 => shapely
        ds_fc_masked = None
        if not os.path.exists(seas51_file):
            print(f"SEA51 file {seas51_file} not found => skip forecast side for {short_label}.")
        else:
            ds_fc = xr.open_dataset(seas51_file)
            if "time" not in ds_fc.coords and "init" in ds_fc.coords:
                ds_fc = ds_fc.rename({"init":"time"})
            if "time" in ds_fc.coords:
                var_fc = "spi3" if "spi3" in ds_fc.data_vars else list(ds_fc.data_vars)[0]
                ds_fc_mam = ds_fc[var_fc].sel(time=ds_fc.time.dt.month.isin([3,4,5])).dropna(dim="time", how="all")
                if ds_fc_mam.time.size == 0:
                    print("No MAM data in SEA51 => skip.")
                else:
                    ds_fc_mam = ds_fc_mam.isel(time=0)
                    for dname in ["lead","member","number"]:
                        if dname in ds_fc_mam.dims and ds_fc_mam.sizes[dname] > 1:
                            ds_fc_mam = ds_fc_mam.isel({dname:0})
                    lat_1d_fc = ds_fc_mam["lat"].values
                    lon_1d_fc = ds_fc_mam["lon"].values
                    lon2d_fc, lat2d_fc = np.meshgrid(lon_1d_fc, lat_1d_fc)
                    mask_fc_2d = shapely.vectorized.contains(union_geom, lon2d_fc, lat2d_fc)
                    ds_fc_masked = ds_fc_mam.where(mask_fc_2d)

        # 3) Plot => row i => left=CHIRPS, right=SEA51
        # Left:
        if ds_ch_masked is not None:
            ds_ch_masked.plot(
                ax=ax_ch, cmap="viridis",
                transform=ccrs.PlateCarree(),
                cbar_kwargs={"label":"CHIRPS SPI3"}
            )
            gdf_union.plot(
                ax=ax_ch, facecolor="none", edgecolor="red",
                linewidth=1.5, transform=ccrs.PlateCarree()
            )
            ax_ch.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
            ax_ch.set_title(f"{short_label.upper()} - CHIRPS MAM => {name_list}")
        else:
            ax_ch.set_title(f"{short_label.upper()} - NO CHIRPS DATA => {name_list}")

        # Right:
        if ds_fc_masked is not None:
            cmap_magma = mpl.cm.get_cmap("magma").copy()
            cmap_magma.set_bad(color="white")
            ds_fc_masked.plot(
                ax=ax_fc,
                cmap=cmap_magma,
                transform=ccrs.PlateCarree(),
                cbar_kwargs={"label":"SEA51 SPI3"}
            )
            gdf_union.plot(
                ax=ax_fc, facecolor="none", edgecolor="red",
                linewidth=1.5, transform=ccrs.PlateCarree()
            )
            ax_fc.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
            ax_fc.set_title(f"{short_label.upper()} - SEA51 MAM => {name_list}")
        else:
            ax_fc.set_title(f"{short_label.upper()} - NO SEA51 DATA => {name_list}")

    plt.suptitle("Multi-Polygon Masking - Single Output for 5 Region Groups", fontsize=14)

    # Approach 1: direct rect
    plt.tight_layout(rect=[0, 0, 1, 0.95])

# Or approach 2: shift suptitle with y
# plt.suptitle("Multi-Polygon Masking ...", y=1.04)

    out_png = "test_regionmask_seas5_chrips_multi.png"
    plt.savefig(out_png, dpi=120)
    plt.show()
    
    print(f"\nSaved => {out_png}")


if __name__ == "__main__":
    main()


