import pandas as pd

from helpers import clean_text, first_of_cst, ignore_infrequent, fix_grammar, count_cst


def preprocess_tabular_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Amount of text
    df['text_en_len'] = df.text_en.str.len().fillna(0).astype(int)
    df['text_es_len'] = df.text_es.str.len().fillna(0).astype(int)
    df['text_es_per'] = df.text_es_len / (df.text_en_len + df.text_es_len)

    # Categorical columns with more than one value: pick first as main
    df['main_open_space'] = df.open_space_kind\
        .map(clean_text)\
        .map(first_of_cst)
    df.main_open_space = ignore_infrequent(df.main_open_space)

    df['main_street_art'] = df.street_art\
        .map(clean_text)\
        .map(first_of_cst)\
        .map(fix_grammar)
    df.main_street_art = ignore_infrequent(df.main_street_art)

    df['main_establishment'] = df.establishment_kind\
        .map(clean_text)\
        .map(first_of_cst)
    df.main_establishment = ignore_infrequent(df.main_establishment, less_frequent_than=5)

    # Categorical columns with more than one value: amount of values
    df['n_establishments'] = df.establishment_kind\
        .map(clean_text)\
        .map(count_cst)\
        .fillna(0)\
        .astype(int)

    df['n_businesses'] = df.business_name\
        .map(clean_text)\
        .map(count_cst)\
        .fillna(0)\
        .astype(int)

    # df['n_nonbusiness_establishments'] = df.n_establishments - df.n_businesses
    # df.n_nonbusiness_establishments[df.n_nonbusiness_establishments < 0] = 0

    df['n_ads'] = df.ad_product\
        .map(clean_text)\
        .map(count_cst)\
        .fillna(0)\
        .astype(int)

    return df
