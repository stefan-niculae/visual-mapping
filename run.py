import os

from geo_processing import preprocess_parcels_geojson
from tabular_processing import preprocess_tabular_data
from image_processing import copy_images, resize_images, build_images_features, compute_greenery
from text_processing import compute_semantic_similarities
from common_processing import filter_common, merge, add_bins, humanize_labels

PARCELS_GEOJSON_PATH = 'input-data/la-parcels.geojson'  # make sure to unzip first
TABULAR_FEATURES_PATH = 'input-data/tabular-features.csv'
RAW_IMAGES_FOLDER = '~/Downloads/Images'

IMAGES_FOLDER = 'input-data/images'
PREPROCESSED_IMAGES_FOLDER = IMAGES_FOLDER + '/preprocessed'
IMAGE_RESIZE_WIDTH = 800
BROWSER_NAVBAR_HEIGHT = 25  # crop browser address bar from the top (set to 0 or None to disable)

TEXT_EMBEDDINGS_FOLDER = 'tf-logs/en-embeddings'
OUTPUT_SPREADSHEETS_FOLDER = 'outputs'


if __name__ == '__main__':
    geo_df = preprocess_parcels_geojson(PARCELS_GEOJSON_PATH)
    tab_df = preprocess_tabular_data(TABULAR_FEATURES_PATH)
    copy_images(relevant_images=tab_df.image_name, source_folder=RAW_IMAGES_FOLDER, destination_folder=IMAGES_FOLDER)
    resize_images(input_folder=IMAGES_FOLDER, preprocessed_folder=PREPROCESSED_IMAGES_FOLDER, new_width=IMAGE_RESIZE_WIDTH, crop_top_px=BROWSER_NAVBAR_HEIGHT)
    img_df = build_images_features(PREPROCESSED_IMAGES_FOLDER)

    geo_df, tab_df, img_df = filter_common(geo_df, tab_df, img_df)

    os.makedirs(OUTPUT_SPREADSHEETS_FOLDER, exist_ok=True)
    img_df = compute_greenery(img_df)
    tab_df = compute_semantic_similarities(tab_df, OUTPUT_SPREADSHEETS_FOLDER, TEXT_EMBEDDINGS_FOLDER)

    feat_df = merge(geo_df, tab_df, img_df)
    feat_df = add_bins(feat_df, OUTPUT_SPREADSHEETS_FOLDER)
    humanize_labels(feat_df)

    feat_df.to_csv(OUTPUT_SPREADSHEETS_FOLDER + '/features.csv', index=False)
