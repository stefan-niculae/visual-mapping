import pandas as pd
import numpy as np
import geopandas as gpd

from text_processing import SIMILARITY_WORDS


def filter_common(geo_df, tab_df, img_df, parcels_geojson_path: str) -> tuple:
    """ only keep entries for which we have (1) an image, (2) parcel info, (3) tabular features """
    tab_parcels = set(tab_df.parcel.astype(str))
    geo_parcels = set(geo_df.parcel.astype(str))
    common_parcels = tab_parcels & geo_parcels

    img_files = set(img_df.image_name)
    tab_files = set(tab_df.image_name)
    common_images = img_files & tab_files

    tab_df_filtered = tab_df[
        tab_df.parcel.isin(common_parcels) &
        tab_df.image_name.isin(common_images)
    ]
    common_parcels = set(tab_df_filtered.parcel)
    common_images  = set(tab_df_filtered.image_name)

    geo_df_filtered = geo_df[geo_df.parcel.isin(common_parcels)]
    img_df_filtered = img_df[img_df.image_name.isin(common_images)]

    # geojson = gpd.read_file(parcels_geojson_path)
    # geojson = geojson[geojson.OBJECTID.astype(str).isin(common_parcels)]

    return geo_df_filtered, tab_df_filtered, img_df_filtered


def merge(geo_df_filtered, tab_df_filtered, img_df_filtered) -> pd.DataFrame:
    merged_df = tab_df_filtered.merge(img_df_filtered, on='image_name')
    geo_df_filtered['parcel_designation'] = geo_df_filtered.zone_summary.str.lower()

    merged_df.parcel = merged_df.parcel.astype(int)
    merged_df = merged_df.merge(geo_df_filtered[['parcel', 'parcel_designation']], on='parcel')
    merged_df['year'] = merged_df.date.str[:4]

    return merged_df\
        .sort_values(by='date')\
        .fillna('n/a')


def _compute_bins(series):
    """
    >>> xs = [-np.inf, -5, -1.2, -.5, .3, 1.5, 7, np.nan]; dict(zip(xs, _compute_bins(xs)))
    {-inf: 'none',
     -5: 'very low',
     -1.2: 'low',
     -0.5: 'med',
     0.3: 'med',
     1.5: 'high',
     7: 'very high',
     nan: nan}
    """
    return pd.cut(
        series,
        bins=[-np.inf, -100, -3, -1, +1, +3, np.inf],
        labels=['none', 'very low', 'low', 'med', 'high', 'very high'],
        include_lowest=True
    )


def add_bins(df, outputs_folder: str) -> pd.DataFrame:
    numeric_df = df._get_numeric_data().drop(columns=['block', 'parcel', 'year'])

    transf_df = numeric_df.applymap(np.log)  # normalize
    transf_df_ = transf_df.replace(-np.inf, np.nan)
    zscores = (transf_df - transf_df_.mean()) / transf_df_.std()

    bins = zscores.apply(_compute_bins).add_suffix('_bin')
    out_df = pd.concat([df, bins], axis=1)

    for col in out_df.columns:
        out_df[col] = out_df[col].astype(str)
    out_df = out_df.replace('nan', 'n/a')

    group_orders = pd.DataFrame({
        col: ['n/a', 'none', 'very low', 'low', 'med', 'high', 'very high']
        for col in bins.columns
    })
    group_orders.to_csv(outputs_folder + '/group_orders.csv', index=False)

    return out_df


def humanize_labels(df) -> pd.DataFrame:
    df['date_formatted'] = pd.to_datetime(df.date).dt.strftime('%b %Y')

    df['Amount of en. text'] = df.text_en_len_bin.replace({
        'none': 'no en. text',
        'very low': 'very little en. text',
        'low': 'little en. text',
        'med': 'some en. text',
        'high': 'much en. text',
        'very high': 'very much en. text',
    })
    df['Amount of es. text'] = df.text_es_len_bin.replace({
        'none': 'no es. text',
        'very low': 'very little es. text',
        'low': 'little es. text',
        'med': 'some es. text',
        'high': 'much es. text',
        'very high': 'very much es. text',
    })

    df['Amount of establishments'] = df.n_establishments_bin.replace({
        'none': 'no establishments',
        'very low': 'very few establishments',
        'low': 'few establishments',
        'med': 'some establishments',
        'high': 'many establishments',
        'very high': 'very many establishments',
    })
    df['Amount of businesses'] = df.n_businesses_bin.replace({
        'none': 'no businesses',
        'very low': 'very few businesses',
        'low': 'few businesses',
        'med': 'some businesses',
        'high': 'many businesses',
        'very high': 'very many businesses',
    })
    df['Amount of ads'] = df.n_ads_bin.replace({
        'none': 'no ads',
        'very low': 'very few ads',
        'low': 'few ads',
        'med': 'some ads',
        'high': 'many ads',
        'very high': 'very many ads',
    })

    for word in SIMILARITY_WORDS:
        w = f'"{word}"'
        df[f'Text closeness to {w}'] = df[f'{w[1:-1]}_SES_bin'].replace({
            'very low': f'very far from {w}',
            'low': f'far from {w}',
            'med': f'somewhat close to {w}',
            'high': f'close to {w}',
            'very high': f'very close to {w}',
        }).fillna('no text')

    df['Colorfulness'] = df.colorfulness_bin.replace({
        'none': 'not colorful at all',
        'very low': 'not colorful',
        'low': 'not too colorful',
        'med': 'somewhat colorful',
        'high': 'colorful',
        'very high': 'very colorful',
    })
    df['Greenness'] = df.greenery_per_bin.replace({
        'none': 'no greenery',
        'very low': 'very little greenery',
        'low': 'little greenry',
        'med': 'some greenery',
        'high': 'much greenery',
        'very high': 'very much greenery',
    })

    df.main_open_space.fillna('not an open space', inplace=True)
    df.main_street_art.fillna('none', inplace=True)
    df.main_establishment.fillna('none', inplace=True)
    df.parcel_designation.fillna('not specified', inplace=True)

    return df
