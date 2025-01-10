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
import geopandas as gpd
import xarray as xr
from vthree_utils import BinCreateParams
from utils_plots import run_map_plot

def main():
    
    load_dotenv()
    ea_input_path = "/Users/sinugp/Downloads/ibf-thresholds-triggers/src/"
    sa_file = "./coiled-data.json"
    polygon_pq_uri = "gs://seas51/ea_admin0_2_custom_polygon_shapefile_v5.parquet"


    # Region-specific settings
    region_name_dict = {82: "Marsabit"}
    region_filter_map = {82: "KEN-ADM1-3_0_0-B32"}  # Filter for Marsabit


    # Seasons and lead times
    seasons = ["MAM"]
    leads = [2, 3, 4]

    # For each region/polygon in region_filter_map:
    for region_id, reg_filter in region_filter_map.items():
        for seas in seasons:
            for lead in leads:
                print(f"\n=== Running run_map_plot() for region_id={region_id}, "
                      f"season={seas}, lead={lead}, filter={reg_filter} ===")

                params = BinCreateParams(
                    region_id=region_id,
                    region_name_dict=region_name_dict,
                    season_str=seas,
                    lead_int=lead,
                    level="mod",                  # or 'sev'/'ext'
                    spi_prod_name="spi3",
                    data_path="",
                    spi4_data_path="",
                    output_path=os.path.join(os.getcwd(), "marsa"),
                    obs_netcdf_file=os.path.join(os.getcwd(), "refined_output.nc"),
                    fct_netcdf_file=os.path.join(os.getcwd(), "jgl_fct_spi3.nc"),
                    service_account_json=sa_file,
                    gcs_file_url=polygon_pq_uri,
                    region_filter=reg_filter,
                    
                )

                # Now we call your normal map-plot function
                run_map_plot(params)

    print("\nAll runs complete.")

if __name__ == "__main__":
    main()

