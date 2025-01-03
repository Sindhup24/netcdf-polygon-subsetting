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

# """
# utils_updated.py
#
# CHANGES IN THIS UPDATED VERSION:
# --------------------------------
# 1. Removed REGION_PATTERN and the .str.contains(...) filter for region_id_pattern.
# 2. Switched from dask.dataframe to pyarrow + pandas for in-memory Parquet reading.
# 3. Added a 'unique_name' column in load_region_mask to avoid Regionmask duplicates.
# 4. Simplified environment variables (removed REGION_PATTERN).
# 5. Added a tqdm progress bar in main_processing_pipeline for stepwise progress.
# """
#

# +
import os
import random
import numpy as np
import pandas as pd
import fsspec
import pyarrow.parquet as pq
import xarray as xr
import xesmf as xe
import regionmask
import geopandas as gpd

from shapely import wkb
from google.oauth2 import service_account
from dotenv import load_dotenv
from tqdm import tqdm

# xclim for SPI
from xclim.indices.stats import standardized_index_fit_params
from xclim.indices import standardized_precipitation_index


# +
###############################################################################
# 1) Environment + Credentials
###############################################################################

def get_env_variables():
    """
    Load environment variables from .env file.

    Required:
      - CREDENTIALS_PATH
      - REGION_PARQUET_PATH
      - CHIRPS_ZARR_PATH
      - SEAS51_ZARR_PATH
      - OUTPUT_PATH

    (Note: We REMOVED 'REGION_PATTERN' since we don't filter on it anymore.)
    """
    required_vars = [
        'CREDENTIALS_PATH',
        'REGION_PARQUET_PATH',
        'CHIRPS_ZARR_PATH',
        'SEAS51_ZARR_PATH',
        'OUTPUT_PATH'
    ]
    
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        env_vars[var.lower()] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env_vars


def setup_gcs_credentials(credentials_path, scope="read_only"):
    """
    Creates GCS credentials for read-only or read-write access.
    """
    scope_dict = {
        "read_only": ["https://www.googleapis.com/auth/devstorage.read_only"],
        "read_write": ["https://www.googleapis.com/auth/devstorage.read_write"]
    }
    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=scope_dict[scope]
    )



# +
###############################################################################
# 2) Load Region Mask (NO region_id_pattern)
###############################################################################

def load_region_mask(parquet_path, credentials):
    """
    Load the entire parquet file (no pattern filter),
    convert geometry from WKB to Shapely,
    and create a Regionmask with unique "names" to avoid duplicates.
    """
    # 1) Read parquet in memory (no Dask)
    fs = fsspec.filesystem("gs", token=credentials)
    with fs.open(parquet_path, "rb") as f:
        table = pq.read_table(f)
    df = table.to_pandas()

    # 2) Convert WKB -> Shapely
    df["geometry"] = df["geometry"].apply(wkb.loads)
    
    # 3) Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # 4) Create a numeric "numbers" column
    gdf["numbers"] = range(len(gdf))

    # 5) Create a guaranteed-unique "unique_name" column
    #    by combining 'gbid' with the row index or "numbers".
    #    This ensures no duplicates, even if "gbid" repeats.
    gdf["unique_name"] = gdf["gbid"] + "_" + gdf.index.astype(str)

    # 6) Create Regionmask - pass "numbers" for region IDs, "unique_name" for names
    regions = regionmask.from_geopandas(
        gdf,
        numbers="numbers",      # unique numeric IDs
        names="unique_name",    # guaranteed-unique string names
        name="custom_regions",
        overlap=False
    )

    return regions, gdf


# +
###############################################################################
# 3) Load + Subset Data
###############################################################################

def load_and_subset_data(zarr_path, credentials, region_mask):
    """
    Load data from Zarr, subset via region_mask.
    """
    fs = fsspec.filesystem("gs", token=credentials)
    mapper = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=False)
    
    mask = region_mask.mask(ds.longitude, ds.latitude)
    ds_masked = ds.where(mask >= 0)
    return ds_masked



# +
###############################################################################
# 4) Regrid Observations
###############################################################################

def regrid_observations(ds, target_resolution=0.25):
    """
    Regrid precipitation from ~5km to ~25km (0.25 deg).
    """
    lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
    lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
    
    ds_out = xr.Dataset({
        "lat": (["lat"], np.arange(lat_min, lat_max + target_resolution, target_resolution)),
        "lon": (["lon"], np.arange(lon_min, lon_max + target_resolution, target_resolution))
    })
    
    ds_renamed = ds.rename({"longitude": "lon", "latitude": "lat"})
    regridder = xe.Regridder(ds_renamed, ds_out, "bilinear", reuse_weights=False)
    regridded_data = regridder(ds_renamed["precip"], keep_attrs=True)
    return regridded_data.to_dataset(name="precip")



# +
###############################################################################
# 5) Calculate SPI
###############################################################################

def calculate_spi(dataset, precip_var='precip', window=3, cal_period=('1991-01-01','2018-01-01')):
    """
    Standardized Precipitation Index using xclim (in memory).
    """
    dataset[precip_var].attrs['units'] = 'mm/month'
    spi = standardized_precipitation_index(
        dataset[precip_var],
        freq="MS",
        window=window,
        dist="gamma",
        method="APP",
        cal_start=cal_period[0],
        cal_end=cal_period[1],
        fitkwargs={"floc": 0}
    )
    return spi.compute()


def calculate_forecast_spi(forecast_ds, lead_times, window=3):
    """
    SPI for forecast data (multiple leads, ensemble members).
    """
    spi_results = []
    for lead in lead_times:
        lead_ds = forecast_ds.sel(forecastMonth=lead)
        member_spis = []
        
        for member in lead_ds.number.values:
            member_ds = lead_ds.sel(number=member)
            spi = calculate_spi(member_ds, precip_var='tprate', window=window)
            member_spis.append(spi)
        
        lead_spi = xr.concat(member_spis, dim='member')
        spi_results.append(lead_spi)
    
    final_spi = xr.concat(spi_results, dim='lead')
    return final_spi.to_dataset(name='spi3')



# +
###############################################################################
# 6) Regrid Forecast
###############################################################################

def regrid_forecast(forecast_spi, obs_dataset):
    """
    Regrid forecast SPI to the same lat/lon as obs.
    """
    regridded_list = []
    
    for lead in forecast_spi.lead.values:
        ds_out = xr.Dataset({
            "lat": (["lat"], obs_dataset.lat.values),
            "lon": (["lon"], obs_dataset.lon.values)
        })
        
        lead_data = forecast_spi.sel(lead=lead).rename({"latitude": "lat", "longitude": "lon"})
        regridder = xe.Regridder(lead_data, ds_out, method="bilinear", reuse_weights=False)
        regridded = regridder(lead_data["spi3"], keep_attrs=True)
        regridded_list.append(regridded.to_dataset())
    
    final_fcst = xr.concat(regridded_list, dim='lead')
    final_fcst = final_fcst.rename({"time": "init"})
    final_fcst["lead"].attrs["units"] = "months"
    return final_fcst


# +
###############################################################################
# 7) Step-by-Step Pipeline
###############################################################################

def main_processing_pipeline():
    """
    Pipeline that does NOT filter the parquet by any region pattern.
    1) Load env + credentials
    2) Load entire parquet -> regionmask
    3) Subset + regrid obs, compute SPI, save
    4) Subset + compute forecast SPI, regrid, save
    """
    # A) Load env vars
    env_path = "/Users/sinugp/Downloads/env"
    load_dotenv(dotenv_path=env_path)
    env_vars = get_env_variables()
    
    # B) We have 7 steps total
    total_steps = 7
    with tqdm(total=total_steps, desc="Pipeline Progress", unit="step") as pbar:
        
        # Step 1: Credentials
        creds = setup_gcs_credentials(env_vars["credentials_path"])
        pbar.set_description("Step 1: GCS creds set up")
        pbar.update(1)
        
        # Step 2: Load region mask from entire parquet (no filtering)
        # main_processing_pipeline code snippet

        # Step 2: Load region mask
        regions, gdf = load_region_mask(env_vars["region_parquet_path"], creds)
        pbar.set_description("Loaded region mask (unique names)")
        pbar.update(1)

        
        # Step 3: Subset + regrid OBS
        obs_ds = load_and_subset_data(env_vars["chirps_zarr_path"], creds, regions)
        obs_ds_25km = regrid_observations(obs_ds)
        pbar.set_description("Step 3: Obs subset + regridded")
        pbar.update(1)
        
        # Step 4: Calculate SPI for obs, save
        obs_spi = calculate_spi(obs_ds_25km)
        obs_out = os.path.join(env_vars["output_path"], "obs_spi3_all.nc")
        obs_spi.to_netcdf(obs_out)
        pbar.set_description(f"Step 4: Obs SPI saved => {obs_out}")
        pbar.update(1)
        
        # Step 5: Subset forecast
        fcst_ds = load_and_subset_data(env_vars["seas51_zarr_path"], creds, regions)
        pbar.set_description("Step 5: Forecast subset loaded")
        pbar.update(1)
        
        # Step 6: Forecast SPI
        fcst_spi = calculate_forecast_spi(fcst_ds, lead_times=range(1,7))
        pbar.set_description("Step 6: Forecast SPI calculated")
        pbar.update(1)
        
        # Step 7: Regrid forecast, save
        regridded_fcst = regrid_forecast(fcst_spi, obs_spi)
        fcst_out = os.path.join(env_vars["output_path"], "forecast_spi3_all.nc")
        regridded_fcst.to_netcdf(fcst_out)
        pbar.set_description(f"Step 7: Forecast SPI regridded => {fcst_out}")
        pbar.update(1)


if __name__ == "__main__":
    main_processing_pipeline()
# -




