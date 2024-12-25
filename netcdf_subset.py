import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask
from shapely import wkb

def decode_wkb_and_create_gdf(csv_file):
    """Decode WKB geometries from CSV and create a GeoDataFrame."""
    df = pd.read_csv(csv_file)
    df['geometry'] = df['geometry'].apply(lambda geom: wkb.loads(eval(geom)) if geom else None)
    df = df.dropna(subset=['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)

    # Make names unique
    if gdf['name'].duplicated().any():
        gdf['name'] = gdf['name'] + "_" + gdf.groupby('name').cumcount().astype(str)
    return gdf

def create_region_mask(gdf, ds):
    """Create a region mask from GeoDataFrame."""
    region_mask = regionmask.from_geopandas(gdf, names="name", numbers="id", overlap=False)
    mask = region_mask.mask(ds['lon'], ds['lat'])
    return mask

def expand_and_broadcast_mask(mask, ds):
    """Expand and broadcast the mask to align with NetCDF dimensions."""
    aligned_mask = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': ds['lat'], 'lon': ds['lon']})
    for dim in ['lead', 'member', 'init']:
        aligned_mask = aligned_mask.expand_dims({dim: ds[dim]})
    expanded_mask = xr.broadcast(aligned_mask, ds)[0]
    return expanded_mask

def subset_netcdf(netcdf_file, csv_file, output_file):
    """Subset NetCDF data using polygons from CSV."""
    # Decode WKB and create GeoDataFrame
    gdf = decode_wkb_and_create_gdf(csv_file)

    # Load NetCDF file
    ds = xr.open_dataset(netcdf_file)

    # Create region mask
    mask = create_region_mask(gdf, ds)

    # Expand and broadcast mask
    expanded_mask = expand_and_broadcast_mask(mask, ds)

    # Subset data
    subset = ds.where(expanded_mask == 1, drop=True)

    # Fix attributes and encoding
    for var in subset.data_vars:
        subset[var].attrs = {}
    encoding = {var: {"_FillValue": None} for var in subset.data_vars}

    # Save subsetted data
    subset.to_netcdf(output_file, encoding=encoding, engine="netcdf4")
    print(f"Subsetted NetCDF file saved successfully at {output_file}")

if __name__ == "__main__":
    netcdf_file = "wjr_fct_spi3.nc"
    csv_file = "ea_admin0_2_custom_polygon_shapefile_v5.csv"
    output_file = "subsetted_data.nc"

    subset_netcdf(netcdf_file, csv_file, output_file)
