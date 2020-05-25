import geopandas as gpd


def preprocess_parcels_geojson(geojson_path: str) -> gpd.GeoDataFrame:

    df = gpd.read_file(geojson_path)

    df['centroid'] = df.geometry.centroid
    df['lon'] = df.centroid.x
    df['lat'] = df.centroid.y

    df = df.rename(columns={
        'OBJECTID': 'parcel',
        'ZONE_SMRY': 'zone_summary',
        'Shape__Are': 'area',
    })

    return df[['parcel', 'area', 'zone_summary', 'lat', 'lon']]
