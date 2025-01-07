from datetime import datetime

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

from climpred import HindcastEnsemble

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


class BinCreateParams:
    def __init__(
        self,
        region_id,
        season_str,
        lead_int,
        level,
        region_name_dict,
        spi_prod_name,
        data_path,
        spi4_data_path,
        output_path,
        obs_netcdf_file,
        fct_netcdf_file,
        service_account_json,
        gcs_file_url,
        region_filter
    ):
        self.region_id = region_id
        self.season_str = season_str
        self.lead_int = lead_int
        self.sc_season_str = season_str.lower()
        self.level = level
        self.region_name_dict = region_name_dict
        self.spi_prod_name = spi_prod_name
        self.data_path = self._ensure_trailing_slash(data_path)
        self.spi4_data_path = self._ensure_trailing_slash(spi4_data_path)
        self.output_path = self._ensure_trailing_slash(output_path)
        self.obs_netcdf_file = obs_netcdf_file
        self.fct_netcdf_file = fct_netcdf_file
        self.service_account_json = service_account_json
        self.gcs_file_url = gcs_file_url
        self.region_filter = region_filter

        # Create necessary directories
        self._create_directories()

    def _ensure_trailing_slash(self, path):
        """Ensure the path ends with a trailing slash."""
        if not path.endswith(os.path.sep):
            return os.path.join(path, '')
        return path

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [self.output_path]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"Directories created/checked: {', '.join(directories)}")

def spi3_prod_name_creator(ds_ens, var_name):
    """
    Convenience function to generate a list of SPI product
    names, such as MAM, so that can be used to filter the
    SPI product from dataframe

    added with method to convert the valid_time in CF format into datetime at
    line 3, which is the format given by climpred valid_time calculation

    Parameters
    ----------
    ds_ens : xarray dataframe
        The data farme with SPI output organized for
        the period 1981-2023.

    Returns
    -------
    spi_prod_list : String list
        List of names with iteration of SPI3 product names such as
        ['JFM','FMA','MAM',......]

    """
    db = pd.DataFrame()
    db["dt"] = ds_ens[var_name].values
    db["dt1"] = db["dt"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # db['dt1']=db['dt'].to_datetimeindex()
    db["month"] = db["dt1"].dt.strftime("%b").astype(str).str[0]
    db["year"] = db["dt1"].dt.strftime("%Y")
    db["spi_prod"] = (
        db.groupby("year")["month"].shift(2)
        + db.groupby("year")["month"].shift(1)
        + db.groupby("year")["month"].shift(0)
    )
    spi_prod_list = db["spi_prod"].tolist()
    return spi_prod_list


def spi4_prod_name_creator(ds_ens, var_name):
    """
    Convenience function to generate a list of SPI product
    names, such as MAM, so that can be used to filter the
    SPI product from dataframe

    added with method to convert the valid_time in CF format into datetime at
    line 3, which is the format given by climpred valid_time calculation

    Parameters
    ----------
    ds_ens : xarray dataframe
        The data farme with SPI output organized for
        the period 1981-2023.

    Returns
    -------
    spi_prod_list : String list
        List of names with iteration of SPI3 product names such as
        ['JFM','FMA','MAM',......]

    """
    db = pd.DataFrame()
    db["dt"] = ds_ens[var_name].values
    db["dt1"] = db["dt"].apply(
        lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second)
    )
    # db['dt1']=db['dt'].to_datetimeindex()
    db["month"] = db["dt1"].dt.strftime("%b").astype(str).str[0]
    db["year"] = db["dt1"].dt.strftime("%Y")
    db["spi_prod"] = (
        db.groupby("year")["month"].shift(3)
        + db.groupby("year")["month"].shift(2)
        + db.groupby("year")["month"].shift(1)
        + db.groupby("year")["month"].shift(0)
    )
    spi_prod_list = db["spi_prod"].tolist()
    return spi_prod_list



def make_obs_fct_dataset(params):
    """
    Prepares observed and forecasted dataset subsets for a specific region, season, and lead time.

    This function loads observed and forecasted datasets based on the season string length (indicating SPI3 or SPI4),
    applies regional masking, selects the data for the given region by its ID, and subsets the data for the specified
    season and lead time. It then aligns the observed dataset time coordinates with the forecasted dataset valid time
    coordinates and returns both datasets.

    Parameters:
    - region_id (int): The identifier for the region of interest.
    - season_str (str): A string representing the season. The length of this string determines whether SPI3 or SPI4
                        datasets are used ('mam', 'jjas', etc. for SPI3, and longer strings for SPI4).
    - lead_int (int): The lead time index for which the forecast dataset is to be subset.

    Returns:
    - obs_data (xarray.DataArray): The subsetted observed data array for the specified region, season, and aligned time coordinates.
    - ens_data (xarray.DataArray): The subsetted forecast data array for the specified region, season, lead time, and aligned time coordinates.

    Notes:
    - The function assumes the existence of a `data_path` variable that specifies the base path to the dataset files.
    - It requires the `xarray` library for data manipulation and assumes specific naming conventions for the dataset files.
    - Regional masking and season-specific processing rely on externally defined functions and naming conventions.
    - The final alignment of observed dataset time coordinates with forecasted dataset valid time coordinates ensures
      comparability between observed and forecasted values for verification purposes.

    Example Usage:
    >>> obs_data, ens_data = make_obs_fct_dataset(1, 'mam', 0)
    >>> print(obs_data)
    >>> print(ens_data)

    This would load the observed and forecasted SPI3 datasets for region 1 during the 'mam' season and subset them
    for lead time index 0, aligning the observed data time coordinates with the forecasted data valid time coordinates.
    """
    try:
        #the_mask, rl_dict, mds1 = gcs_paraquet_mask_creator(params)
        #bounds = mds1.bounds
        #llon, llat = bounds.iloc[params.region_id][["minx", "miny"]]
        #ulon, ulat = bounds.iloc[params.region_id][["maxx", "maxy"]]

        #logger.debug(
        #    f"Region bounds: llon={llon}, llat={llat}, ulon={ulon}, ulat={ulat}"
        #)

        if len(params.season_str) == 3:
            kn_obs = xr.open_dataset('./kmj-25km-chirps-v2.0.monthly.nc')
            kn_fct = xr.open_dataset('./kn_fct_spi3.nc')
            #logger.info("Loaded SPI3 datasets")
        else:
            kn_fct = xr.open_dataset(os.path.join(params.data_path, params.fct_netcdf_file))
            kn_obs = xr.open_dataset(os.path.join(params.data_path, params.obs_netcdf_file))
            #logger.info("Loaded SPI4 datasets")

        #a_fc = kn_fct.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
        #a_obs = kn_obs.sel(lon=slice(llon, ulon), lat=slice(llat, ulat))
        a_obs=kn_obs
        a_fc=kn_fct
        print("subsetted obs and fcst to given region")
        print("Created HindcastEnsemble")
        hindcast = HindcastEnsemble(a_fc)
        hindcast = hindcast.add_observations(a_obs)

        a_fc1 = hindcast.get_initialized()
        print("Added climpred HindcastEnsemble to add valid_time in fcst")
        a_fc2 = a_fc1.isel(lead=params.lead_int)

        if len(params.season_str) == 3:
            spi_prod_list = spi3_prod_name_creator(a_fc2, "valid_time")
            obs_spi_prod_list = spi3_prod_name_creator(a_obs, "time")
        else:
            spi_prod_list = spi4_prod_name_creator(a_fc2, "valid_time")
            obs_spi_prod_list = spi4_prod_name_creator(a_obs, "time")
        print(
            f"added SPI prodcut in obs and fcst dataset, filtered to {params.season_str}"
        )
        a_fc2 = a_fc2.assign_coords(spi_prod=("init", spi_prod_list))
        a_fc3 = a_fc2.where(a_fc2.spi_prod == params.season_str, drop=True)

        a_obs1 = a_obs.assign_coords(spi_prod=("time", obs_spi_prod_list))
        a_obs2 = a_obs1.where(a_obs1.spi_prod == params.season_str, drop=True)

        # Convert valid_time to numpy datetime64 for comparison from cftime of a_fc3
        fct_valid_times = np.array(
            [np.datetime64(vt.isoformat()) for vt in a_fc3.valid_time.values]
        )
        obs_times = a_obs2.time.values

        # Find common dates
        common_dates = np.intersect1d(fct_valid_times, obs_times)

        # common_dates = np.unique(a_fc3.valid_time.values.ravel())
        # Filter both datasets to include only common dates
        # a_fc4 = a_fc3.sel(valid_time=common_dates)
        a_obs3 = a_obs2.sel(time=common_dates)
        # a_obs3 = a_obs2.sel(time=common_dates)
        a_fc3_init_dates = common_dates.astype("datetime64[M]") - np.timedelta64(
            int(a_fc3.lead.values), "M"
        )
        a_fc4 = a_fc3.sel(init=a_fc3_init_dates)
        # Ensure the time dimension in a_fc4 matches the valid_time coordinate
        # a_fc4 = a_fc4.assign_coords(time=('valid_time', common_dates))
        # a_fc4 = a_fc4.swap_dims({'valid_time': 'time'})

        print(
            f"Found {len(common_dates)} common dates between observed and forecast data"
        )
        print("Successfully prepared observed and forecasted datasets")

        return a_obs3, a_fc4

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except ValueError as e:
        print(f"Value error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in make_obs_fct_dataset: {e}")
        raise
    # return a_obs3, a_fc3


