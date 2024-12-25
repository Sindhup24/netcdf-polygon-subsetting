import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import regionmask
from shapely import wkb

# Step 1: Load GeoDataFrame
csv_file = "/Users/sinugp/Downloads/ea_admin0_2_custom_polygon_shapefile_v5.csv"
df = pd.read_csv(csv_file)

# Decode WKB geometries
def decode_wkb(geometry):
    try:
        return wkb.loads(eval(geometry))
    except:
        return None

df['geometry'] = df['geometry'].apply(decode_wkb)
df = df.dropna(subset=['geometry'])

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)

# Fix duplicate names
if gdf['name'].duplicated().any():
    gdf['name'] = gdf['name'] + "_" + gdf.groupby('name').cumcount().astype(str)

# Step 2: Load NetCDF File
netcdf_file = "/Users/sinugp/Downloads/ibf-thresholds-triggers/src/wjr_fct_spi3.nc"
ds = xr.open_dataset(netcdf_file)

# Step 3: Create Region Mask
region_mask = regionmask.from_geopandas(gdf, names="name", numbers="id", overlap=False)
mask = region_mask.mask(ds['lon'], ds['lat'])

# Step 4: Align Mask with All Dimensions
aligned_mask = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': ds['lat'], 'lon': ds['lon']})
for dim in ['lead', 'member', 'init']:
    aligned_mask = aligned_mask.expand_dims({dim: ds[dim]})
aligned_mask = xr.broadcast(aligned_mask, ds)[0]

# Step 5: Apply the Mask and Subset the Data
subset = ds.where(aligned_mask == 1, drop=True)

# Step 6: Debug and Fix Attributes
# Remove attributes from `lead`
if 'lead' in subset.variables:
    subset['lead'].attrs = {}

# Define encoding to remove `_FillValue`
encoding = {var: {"_FillValue": None} for var in subset.data_vars}

# Step 7: Save the Subsetted File
output_file = "/Users/sinugp/Downloads/subsetted_file.nc"
subset.to_netcdf(output_file, encoding=encoding, engine="netcdf4")

print(f"Subsetted NetCDF file saved successfully at {output_file}")
