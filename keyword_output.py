"""
File which outputs keywords into a csv for demonstration purposes

Python version: 3.7.3
Input: folder containing pitchdecks, name of the desired pitchdeck
Output: a csv file with keywords for the desired pitchdeck
"""
import sys

sys.path.append('C:\\dev\\code\\etap-platform\\etap')

import pandas as pitch_deck
import time
from tfidf import tf_idf_rank

from pitch_deck_collection import PitchDeckCollection


# Path to folder containing the pitchdecks
folder = 'C:\\dev\\data\\ETAP\\pitch_decks\\english'

# Name of the pitchdeck to analize (without extension)
pd_name = 'fastned'

# Min and max ngram count
min_ngram, max_ngram = 1, 4

# Path to save the csv's
save_path = 'C:\\dev\\outputs\\'

# Separator
separator = ';'

# Stemming boolean
stemming = False


# Current time millis function
def current_milli_time():
    return int(round(time.time() * 1000))

########################################################################################################################


collection = PitchDeckCollection(folder, min_ngram=min_ngram, max_ngram=max_ngram)
pdeck = collection.docs[pd_name]

# Collect ngrams
ngrams = pdeck.ngrams
# ngrams_stemmed = None
# if stemming:
#     ngrams_stemmed = hardt_pd.ngrams_stemmed

# Collect non-stemmed tf_idf
rank = tf_idf_rank(ngrams, [doc.ngrams for doc in collection.docs.values()])
# rank_stemmed = None
# if ngrams_stemmed:
#     rank_stemmed = collection.tf_idf_rank(pd_name, stemming=True)

# Create dataframes
d = {'ngram': [], 'length': [], 'count': [], 'tf_idf_rank': []}
for length in ngrams:
    for ngram in ngrams[length]:
        d['ngram'].append(' '.join(ngram))
        d['length'].append(length)
        d['count'].append(ngrams[length][ngram])
        d['tf_idf_rank'].append(rank[length][ngram])
df = pitch_deck.DataFrame(data=d)
df.to_csv(save_path + pd_name + '_keywords.csv', index=False, sep=separator)

# if rank_stemmed:
#     d_stemmed = {'ngram': [], 'length': [], 'count': [], 'tf_idf_rank': []}
#     for length in ngrams_stemmed:
#         for ngram in ngrams_stemmed[length]:
#             d_stemmed['ngram'].append(' '.join(ngram))
#             d_stemmed['length'].append(length)
#             d_stemmed['count'].append(ngrams_stemmed[length][ngram])
#             d_stemmed['tf_idf_rank'].append(rank_stemmed[length][ngram])
#     df_stemmed = pd.DataFrame(data=d_stemmed)
#     df_stemmed.to_csv(save_path + pd_name + '_keywords_stemmed.csv', index=False, sep=separator)
#%%
def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """PageRank: The trillion dollar algorithm.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v