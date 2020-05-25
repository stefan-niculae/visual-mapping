import os
import shutil
import requests
from pathlib import Path

import pandas as pd
from PIL import Image
import cv2

SEGMENTATION_ENDPOINT = 'http://scenesegmentation.csail.mit.edu/cgi-bin/image_segnet.py'
GREENERY_CATEGORIES = [
    'tree',
    'grass',
    'plant',
    'palm'
]


def copy_images(relevant_images: [str], source_folder: str, destination_folder: str):
    os.makedirs(destination_folder, exist_ok=True)
    for name in relevant_images:
        file_name = name.replace('star', '').strip() + '.png'
        try:
            shutil.copy(source_folder      + '/' + file_name,
                        destination_folder + '/' + file_name)
        except FileNotFoundError as e:
            print(e)


def resize_images(input_folder: str, preprocessed_folder: str, new_width: int, crop_top_px: int):
    """ takes about 5 min for 1.5k images """
    os.makedirs(preprocessed_folder, exist_ok=True)

    # resize
    for item in os.listdir(input_folder):
        if not item.endswith('.png'):
            continue
        image_path = input_folder + '/' + item
        image_name, extension = os.path.splitext(item)

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        aspect_ratio = width / height
        new_height = int(1 / aspect_ratio * new_width)

        resized = image.resize((new_width, new_height), Image.ANTIALIAS)
        resized.save(f'{preprocessed_folder}/{image_name}.jpg', format='jpeg', quality=80)

    # crop navigation bar at the top
    if not crop_top_px:
        return
    for item in os.listdir(preprocessed_folder):
        if not item.endswith('.jpg'):
            continue
        image_path = preprocessed_folder + '/' + item

        image = Image.open(image_path)
        width, height = image.size

        top = crop_top_px
        bottom = height
        left = 0
        right = width

        cropped = image.crop((left, top, right, bottom))
        cropped.save(image_path)


def _compute_colorfulness(image_path: str) -> float:
    """ source: https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/ """
    image = cv2.imread(str(image_path))
    (B, G, R) = cv2.split(image.astype(float))

    rg = abs(R - G)
    yb = abs(.5 * (R + G) - B)

    s = (rg.std()  ** 2 + yb.std()  ** 2) ** .5
    m = (rg.mean() ** 2 + yb.mean() ** 2) ** .5

    return (.3 * m) + s


def build_images_features(preprocessed_images_folder: str) -> pd.DataFrame:
    image_paths = [file_path
                   for file_path in Path(preprocessed_images_folder).iterdir()
                   if file_path.name != '.DS_Store']

    return pd.DataFrame({
        'image_name':   map(lambda p: p.stem, image_paths),
        'colorfulness': map(_compute_colorfulness, image_paths),
    })


def _get_segmentation(image_path) -> {str: float}:
    try:
        image_file = open(image_path, 'rb')
        response = requests.post(url=SEGMENTATION_ENDPOINT, files={'data': image_file})

        d = response.json()['object']
        classes = d['classes']
        ratios = map(lambda s: float(s[:-1]) / 100, d['ratios'])

        return dict(zip(classes, ratios))
    except Exception as e:
        return {'error': e}


def compute_greenery(img_df) -> pd.DataFrame:
    """ fetching takes about 1.5h for 1.3k images """
    image_paths = img_df.image_name
    seg_df = pd.DataFrame(
        data=map(_get_segmentation, image_paths),
        index=map(lambda p: p.stem, image_paths)
    ).fillna(0)
    seg_df.index.name = 'image_name'
    seg_df['greenery_per'] = sum(seg_df[categ] for categ in GREENERY_CATEGORIES)

    seg_df = seg_df.reset_index()[['image_name', 'greenery_per']]
    return pd.merge(img_df, seg_df, on='image_name')
