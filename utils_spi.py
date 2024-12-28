import pandas as pd
import fsspec
from google.oauth2 import service_account
import dask.dataframe as dd
import xarray as xr
import geopandas as gpd
from xclim.indices.stats import standardized_index_fit_params
from xclim.indices import standardized_precipitation_index
import xesmf as xe
import numpy as np
import regionmask
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_env_variables():
    """
    Get environment variables from .env file.
    
    Returns:
    --------
    dict
        Dictionary containing environment variables
    """
    required_vars = [
        'CREDENTIALS_PATH',
        'REGION_PARQUET_PATH',
        'CHIRPS_ZARR_PATH',
        'SEAS51_ZARR_PATH',
        'REGION_PATTERN',
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
    Set up Google Cloud Storage credentials.
    
    Parameters:
    -----------
    credentials_path : str
        Path to the service account JSON file
    scope : str
        Type of access scope ("read_only" or "read_write")
        
    Returns:
    --------
    credentials : google.oauth2.service_account.Credentials
        GCS credentials object
    """
    scope_dict = {
        "read_only": ["https://www.googleapis.com/auth/devstorage.read_only"],
        "read_write": ["https://www.googleapis.com/auth/devstorage.read_write"]
    }
    
    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=scope_dict[scope]
    )

def load_region_mask(parquet_path, credentials, region_id_pattern):
    """
    Load and create region mask from parquet file.
    
    Parameters:
    -----------
    parquet_path : str
        GCS path to parquet file containing region geometries
    credentials : google.oauth2.service_account.Credentials
        GCS credentials
    region_id_pattern : str
        Pattern to filter regions (e.g., 'kmj' for specific regions)
        
    Returns:
    --------
    region_mask : regionmask.Regions
        Region mask object for subsetting data
    """
    ddf = dd.read_parquet(
        parquet_path, 
        storage_options={'token': credentials}, 
        engine='pyarrow'
    )
    
    fdf = ddf[ddf['gbid'].str.contains(region_id_pattern)]
    df = fdf.compute()
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    regions = regionmask.from_geopandas(
        gdf,
        numbers=range(len(gdf)),
        names=gdf.gbid.values,
        name="custom_regions"
    )
    
    return regions, gdf

def load_and_subset_data(zarr_path, credentials, region_mask, gdf):
    """
    Load and subset data using region mask.
    
    Parameters:
    -----------
    zarr_path : str
        Path to Zarr dataset
    credentials : google.oauth2.service_account.Credentials
        GCS credentials
    region_mask : regionmask.Regions
        Region mask for subsetting
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing region geometries
        
    Returns:
    --------
    xarray.Dataset
        Subsetted dataset
    """
    fs = fsspec.filesystem("gs", token=credentials)
    mapper = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=False)
    
    mask = region_mask.mask(ds.longitude, ds.latitude)
    ds_masked = ds.where(mask >= 0)
    
    return ds_masked

def regrid_observations(ds, target_resolution=0.25):
    """
    Regrid observations from 5km to 25km resolution.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset at original resolution
    target_resolution : float
        Target resolution in degrees (default 0.25 for 25km)
        
    Returns:
    --------
    xarray.Dataset
        Regridded dataset
    """
    # Determine target grid coordinates
    lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
    lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
    
    # Create target grid
    ds_out = xr.Dataset({
        "lat": (["lat"], 
                np.arange(lat_min, lat_max + target_resolution, target_resolution),
                {"units": "degrees_north"}),
        "lon": (["lon"], 
                np.arange(lon_min, lon_max + target_resolution, target_resolution),
                {"units": "degrees_east"}),
    })
    
    # Rename coordinates to match regridder requirements
    ds_renamed = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    
    # Create regridder
    regridder = xe.Regridder(ds_renamed, ds_out, "bilinear")
    
    # Apply regridding to precipitation data
    regridded_data = regridder(ds_renamed['precip'], keep_attrs=True)
    
    return regridded_data.to_dataset(name='precip')

def calculate_spi(dataset, precip_var='precip', window=3, cal_period=('1991-01-01', '2018-01-01')):
    """
    Calculate Standardized Precipitation Index.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset containing precipitation data
    precip_var : str
        Name of precipitation variable in dataset
    window : int
        Window size for SPI calculation (months)
    cal_period : tuple
        Calibration period (start_date, end_date)
        
    Returns:
    --------
    xarray.DataArray
        Calculated SPI values
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
    Calculate SPI for forecast data across different lead times.
    
    Parameters:
    -----------
    forecast_ds : xarray.Dataset
        Forecast dataset
    lead_times : list
        List of lead times to process
    window : int
        SPI calculation window
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing SPI calculations for all lead times
    """
    spi_results = []
    
    for lead in lead_times:
        lead_ds = forecast_ds.sel(forecastMonth=lead)
        member_spis = []
        
        for member in lead_ds.number.values:
            member_ds = lead_ds.sel(number=member)
            spi = calculate_spi(
                member_ds, 
                precip_var='tprate',
                window=window
            )
            member_spis.append(spi)
        
        lead_spi = xr.concat(member_spis, dim='member')
        spi_results.append(lead_spi)
    
    final_spi = xr.concat(spi_results, dim='lead')
    return final_spi.to_dataset(name='spi3')

def regrid_forecast(forecast_spi, obs_dataset):
    """
    Regrid forecast SPI to match observations grid.
    
    Parameters:
    -----------
    forecast_spi : xarray.Dataset
        Forecast SPI dataset
    obs_dataset : xarray.Dataset
        Observation dataset with target grid
        
    Returns:
    --------
    xarray.Dataset
        Regridded forecast dataset
    """
    regridded_results = []
    
    for lead in forecast_spi.lead.values:
        ds_out = xr.Dataset({
            "lat": (["lat"], obs_dataset.lat.values, {"units": "degrees_north"}),
            "lon": (["lon"], obs_dataset.lon.values, {"units": "degrees_east"}),
        })
        
        lead_data = forecast_spi.sel(lead=lead)
        lead_data = lead_data.rename({'longitude': 'lon', 'latitude': 'lat'})
        
        regridder = xe.Regridder(lead_data, ds_out, "bilinear")
        regridded = regridder(lead_data.spi3, keep_attrs=True)
        
        regridded_results.append(regridded.to_dataset())
    
    final_forecast = xr.concat(regridded_results, dim='lead')
    final_forecast = final_forecast.rename({'time': 'init'})
    final_forecast['lead'].attrs['units'] = 'months'
    
    return final_forecast

def main_processing_pipeline():
    """
    Main processing pipeline to generate SPI forecasts and observations.
    Uses environment variables for configuration.
    """
    # Load environment variables
    env_vars = get_env_variables()
    
    # Setup credentials
    creds = setup_gcs_credentials(env_vars['credentials_path'])
    
    # Load region mask
    regions, gdf = load_region_mask(
        env_vars['region_parquet_path'], 
        creds, 
        env_vars['region_pattern']
    )
    
    # Process observations
    obs_ds = load_and_subset_data(
        env_vars['chirps_zarr_path'], 
        creds, 
        regions, 
        gdf
    )
    
    # Regrid observations to 25km
    obs_ds_25km = regrid_observations(obs_ds)
    
    # Calculate SPI for regridded observations
    obs_spi = calculate_spi(obs_ds_25km)
    obs_spi.to_netcdf(f"{env_vars['output_path']}_obs_spi3.nc")
    
    # Process forecasts
    forecast_ds = load_and_subset_data(
        env_vars['seas51_zarr_path'], 
        creds, 
        regions, 
        gdf
    )
    forecast_spi = calculate_forecast_spi(forecast_ds, lead_times=range(1, 7))
    
    # Regrid forecasts to match observations
    regridded_forecast = regrid_forecast(forecast_spi, obs_spi)
    regridded_forecast.to_netcdf(f"{env_vars['output_path']}_forecast_spi3.nc")

if __name__ == "__main__":
    main_processing_pipeline()
