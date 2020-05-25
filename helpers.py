import re

import numpy as np


def maybe_wrt_na(func):
    def thunk(*args):
        if args[0] is np.nan:  # !! is instead of pd.isna
            return np.nan
        return func(*args)
    return thunk


@maybe_wrt_na
def clean_text(s):
    s = s.strip()
    s = s.rstrip(',')
    if s == '':
        return np.nan
    return s.lower()


@maybe_wrt_na
def split_into_words(s):
    return re.findall('\w+', s)


@maybe_wrt_na
def first_of_cst(s):
    return s.split(', ')[0]


@maybe_wrt_na
def count_cst(s):
    return s.count(',') + 1


@maybe_wrt_na
def fix_grammar(s):
    groups = {
        ('handpainted', 'hand painted', 'hnadpainted'): 'hand-painted',
        ('signs', 'sign'): 'sign(s)',
        ('graffiti',): 'grafitti',
    }

    for ks, v in groups.items():
        for k in ks:
            if v in s:
                continue  # to the next group
            s = s.replace(k, v)

    return s


def ignore_infrequent(series, less_frequent_than=3):
    counts = series.value_counts()
    values_to_ignore = counts[counts < less_frequent_than].index
    ignore_mask = series.isin(values_to_ignore)

    s_ = series.copy()
    s_[ignore_mask] = np.nan
    return s_


