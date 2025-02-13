import os
import time
import uuid
import random
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

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
from dotenv import load_dotenv
from google.oauth2 import service_account
from markupsafe import escape  # For safe error message conversion

# ------------------------------
# Geospatial Processing Function
# ------------------------------

def process_kmj():
    """
    Process geospatial data for the KMJ region and generate a plot with dynamic info.
    """
    # Load environment variables from your .env file
    load_dotenv("/Users/sinugp/Downloads/env")
    
    # Retrieve environment variables
    parquet_url = os.getenv("REGION_PARQUET_PATH")
    service_account_json = os.getenv("CREDENTIALS_PATH")
    if not parquet_url or not service_account_json:
        raise ValueError("Missing REGION_PARQUET_PATH or CREDENTIALS_PATH in .env")

    # Set file paths for the datasets
    chirps_file = "kmj-25km-chirps-v2.0.monthly.nc"
    seas51_file = "kn_fct_spi3.nc"

    # Create credentials for accessing the Parquet file in GCS
    creds = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )

    print(f"[{datetime.datetime.now()}] Reading KMJ polygon from {parquet_url} ...")
    ddf = dd.read_parquet(
        parquet_url,
        engine="pyarrow",
        storage_options={"token": creds}
    )
    fdf = ddf[ddf["gbid"].str.contains("kmj", case=False, na=False)]
    df_local = fdf.compute()
    if df_local.empty:
        raise ValueError("No 'kmj' found in the Parquet file.")

    # Convert the WKB geometry to a shapely geometry
    df_local["geometry"] = df_local["geometry"].apply(shapely.wkb.loads)
    gdf_local = gpd.GeoDataFrame(df_local, geometry="geometry")
    kmj_geom = gdf_local.iloc[0].geometry
    print("Retrieved kmj_geom (no buffer).")

    # Create a GeoDataFrame for plotting the polygon
    gdf_kmj = gpd.GeoDataFrame(geometry=[kmj_geom], crs=gdf_local.crs)

    # Open the CHIRPS dataset and select MAM months (March, April, May)
    ds_ch = xr.open_dataset(chirps_file)
    var_ch = "spi3" if "spi3" in ds_ch.data_vars else "precip"
    ds_ch_mam = ds_ch[var_ch].sel(time=ds_ch.time.dt.month.isin([3, 4, 5])).dropna(dim="time", how="all")
    ds_ch_mam = ds_ch_mam.isel(time=0)  # Select the first time slice
    lat_1d = ds_ch_mam["lat"].values
    lon_1d = ds_ch_mam["lon"].values
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    mask_ch_2d = shapely.vectorized.contains(kmj_geom, lon2d, lat2d)
    ds_ch_masked = ds_ch_mam.where(mask_ch_2d)

    # Process SEA51 dataset if it exists
    ds_fc_masked = None
    if os.path.exists(seas51_file):
        ds_fc = xr.open_dataset(seas51_file)
        if "time" not in ds_fc.coords and "init" in ds_fc.coords:
            ds_fc = ds_fc.rename({"init": "time"})
        if "time" in ds_fc.coords:
            var_fc = "spi3" if "spi3" in ds_fc.data_vars else list(ds_fc.data_vars)[0]
            ds_fc_mam = ds_fc[var_fc].sel(time=ds_fc.time.dt.month.isin([3, 4, 5])).dropna(dim="time", how="all")
            ds_fc_mam = ds_fc_mam.isel(time=0)
            for dname in ["lead", "member", "number"]:
                if dname in ds_fc_mam.dims and ds_fc_mam.sizes[dname] > 1:
                    ds_fc_mam = ds_fc_mam.isel({dname: 0})
            lat_1d_fc = ds_fc_mam["lat"].values
            lon_1d_fc = ds_fc_mam["lon"].values
            lon2d_fc, lat2d_fc = np.meshgrid(lon_1d_fc, lat_1d_fc)
            mask_fc_2d = shapely.vectorized.contains(kmj_geom, lon2d_fc, lat2d_fc)
            ds_fc_masked = ds_fc_mam.where(mask_fc_2d)

    # Create side-by-side plots for CHIRPS and SEA51 datasets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    ax0, ax1 = axes

    ds_ch_masked.plot(
        ax=ax0, cmap="viridis",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": f"{var_ch} (CHIRPS)"}
    )
    minx, miny, maxx, maxy = kmj_geom.bounds
    gdf_kmj.plot(ax=ax0, facecolor="none", edgecolor="red", linewidth=1.5, transform=ccrs.PlateCarree())
    ax0.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
    ax0.set_title("CHIRPS MAM (exact polygon)")

    if ds_fc_masked is not None:
        cmap_magma = mpl.cm.get_cmap("magma").copy()
        cmap_magma.set_bad(color="white")
        ds_fc_masked.plot(
            ax=ax1,
            cmap=cmap_magma,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={"label": "SEA51"}
        )
        gdf_kmj.plot(ax=ax1, facecolor="none", edgecolor="red", linewidth=1.5, transform=ccrs.PlateCarree())
        ax1.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
        ax1.set_title("SEA51 MAM (exact polygon)")
    else:
        ax1.set_title("SEA51 => none found")

    # Add dynamic title information with current time and a random value.
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rand_value = random.randint(1, 1000)
    plt.suptitle(f"KMJ - MAM - CHIRPS VS SEA51\nProcessed at {current_time} | Random: {rand_value}")

    plt.tight_layout()
    output_path = "/Users/sinugp/Downloads/test_regionmask_seas5_chrips.png"
    plt.savefig(output_path, dpi=120)
    plt.close()

    print(f"Plot saved to: {output_path}")
    return "Workflow completed successfully."

# ------------------------------
# Helper Classes and Functions for Triggers
# ------------------------------

# Default configuration for triggers
DEFAULT_CONFIG = {
    'trigger_file': 'trigger_file.txt',
    'trigger_interval': 5  # seconds between time-based triggers
}

class ProcessingState:
    """
    A simple processing state to ensure that only one trigger runs the process at a time.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.processing = False

    def start_processing(self):
        if self._lock.acquire(blocking=False):
            if not self.processing:
                self.processing = True
                return True
            self._lock.release()
        return False

    def end_processing(self):
        self.processing = False
        self._lock.release()

def process_with_timeout(executor, timeout=60):
    """
    Submit the process_kmj task to the executor and wait for it to finish with a timeout.
    """
    future = executor.submit(process_kmj)
    try:
        result = future.result(timeout=timeout)
        print(result)
    except Exception as e:
        print(f"Error during processing: {escape(str(e))}")

# ------------------------------
# Trigger System Class
# ------------------------------

class TriggerSystem:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.state = ProcessingState()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = threading.Event()
        
    def start_file_monitoring(self):
        """Monitor for file-based triggers."""
        while not self._stop_event.is_set():
            if os.path.exists(self.config['trigger_file']):
                print(f"File trigger detected: {self.config['trigger_file']}")
                if self.state.start_processing():
                    try:
                        process_with_timeout(self.executor)
                    finally:
                        self.state.end_processing()
                        os.remove(self.config['trigger_file'])
                        print(f"Trigger file {self.config['trigger_file']} removed after processing.")
            time.sleep(1)
    
    def start_time_based_trigger(self):
        """Run time-based triggers."""
        while not self._stop_event.is_set():
            print("Time trigger event detected.")
            if self.state.start_processing():
                try:
                    process_with_timeout(self.executor)
                finally:
                    self.state.end_processing()
            time.sleep(self.config['trigger_interval'])
    
    def start(self):
        """Start both trigger types in separate threads."""
        threads = [
            threading.Thread(target=self.start_file_monitoring, daemon=True),
            threading.Thread(target=self.start_time_based_trigger, daemon=True)
        ]
        for thread in threads:
            thread.start()
        return threads
    
    def stop(self):
        """Stop all trigger monitoring."""
        self._stop_event.set()
        self.executor.shutdown(wait=True)
        print("Trigger system stopped.")

# ------------------------------
# Main Function to Run the Trigger System
# ------------------------------

def main():
    # Create a trigger system instance
    trigger_system = TriggerSystem()

    # Start the trigger system threads
    threads = trigger_system.start()
    
    # For demonstration purposes, let the trigger system run for 30 seconds.
    # Meanwhile, we can simulate a file trigger by creating the trigger file.
    print("Trigger system running. You can simulate a file trigger by creating 'trigger_file.txt'.")
    
    # Simulate file trigger after 10 seconds:
    time.sleep(10)
    with open(DEFAULT_CONFIG['trigger_file'], "w") as f:
        f.write(str(uuid.uuid4()))
    print(f"Simulated file trigger created: {DEFAULT_CONFIG['trigger_file']}")
    
    # Let the system run for additional 20 seconds
    time.sleep(20)
    
    # Stop the trigger system
    trigger_system.stop()
    
    # Join threads (optional if you want to wait for them to finish cleanly)
    for thread in threads:
        thread.join()
    
    print("Main function complete.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {escape(str(e))}")
