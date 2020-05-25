import os
from collections import Counter

import numpy as np
import pandas as pd
import gensim.downloader as api
import scipy
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
from tensorboard.plugins import projector

from helpers import clean_text, split_into_words, maybe_wrt_na

EMBEDDING_MODEL = 'word2vec-google-news-300'
SIMILARITY_WORDS = [
    'liquor',
    'food',
    'money',
    'health',
    'car',
]


@maybe_wrt_na
def _sum_vec(wv, ws):
    return sum(wv[w] for w in ws if w in wv)


@maybe_wrt_na
def similarity(v, other):
    return 1 - scipy.spatial.distance.cosine(v, other)


def compute_semantic_similarities(tab_df, outputs_folder: str, embeddings_log_dir: str):
    wv = api.load(EMBEDDING_MODEL)  # takes about 2 min to load from disk, and a lot more to download the first time

    entry_words = tab_df.text_en\
        .map(clean_text)\
        .map(split_into_words)
    entry_vec = entry_words.map(_sum_vec)

    # Projecting
    all_words = sum([ws for ws in entry_words if type(ws) is list], [])
    word_counts = Counter(all_words)
    unique_words = list(set(all_words))
    known_words = [w for w in unique_words if w in wv]
    frequent_words = [w for w in known_words if word_counts[w] >= 5]

    # 2D visualization
    vecs = np.array([wv[w] for w in frequent_words])
    vecs -= vecs.mean()  # spherize
    vecs /= vecs.std()
    points = UMAP().fit_transform(vecs)

    clusters = AgglomerativeClustering(n_clusters=8).fit(points)
    viz_df = pd.DataFrame({
        'word': frequent_words,
        'dim1': points[:, 0],
        'dim2': points[:, 1],
        'cluster': map(lambda label: chr(ord('A') +label), clusters.labels_),
    })
    viz_df.to_csv(outputs_folder + '/word-embedding-projections.csv', index=False)

    # Tensorboard visualization
    os.makedirs(embeddings_log_dir, exist_ok=True)
    with open(embeddings_log_dir + '/metadata.tsv', 'w') as f:
        f.write('\n'.join(known_words))

    known_vecs = tf.Variable([wv[w] for w in known_words])
    checkpoint = tf.train.Checkpoint(embedding=known_vecs)
    checkpoint.save(embeddings_log_dir + '/embedding.ckpt')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(embeddings_log_dir, config)
    print('To visualize word embeddings: first run tensorboard --logdir', embeddings_log_dir,
          'then visit http://localhost:6006/#projector')

    # Similarity
    similarity_df = pd.DataFrame({
        term + '_SES': entry_vec.map(lambda v: similarity(v, wv[term]))
        for term in SIMILARITY_WORDS
    })
    similarity_df['image_name'] = tab_df.image_name

    return tab_df.merge(similarity_df, on='image_name')
