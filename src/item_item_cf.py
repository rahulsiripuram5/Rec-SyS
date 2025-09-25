import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import load_movie_titles



# Load dataset
p = "data/ml-100k/u.data"
df = pd.read_csv(p, sep='\t', names=['user_id','item_id','rating','timestamp'])

# Create user–item rating matrix (rows = users, cols = items)
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

# Fill NaNs with 0 (unrated = 0 for similarity computation)
item_user_matrix = user_item_matrix.T.fillna(0)

# Compute cosine similarity between items
similarity = cosine_similarity(item_user_matrix)
similarity_df = pd.DataFrame(similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

def recommend_items(user_id, top_k=5):
    """Recommend top_k items for a given user based on item-item CF"""
    user_ratings = user_item_matrix.loc[user_id].dropna()
    already_rated = user_ratings.index.tolist()

    scores = {}
    for item, rating in user_ratings.items():
        sims = similarity_df[item].drop(already_rated)  # ignore already rated
        for sim_item, sim_score in sims.items():
            scores[sim_item] = scores.get(sim_item, 0) + sim_score * rating

    # Sort and return top_k items
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in ranked[:top_k]]

# Example: recommend for user 405
# Load movie titles
movie_dict = load_movie_titles()

# Example: recommend for user 405
recs = recommend_items(405, top_k=5)
print("Recommendations for user 405 (IDs):", recs)
print("Recommendations for user 405 (Titles):", [movie_dict[i] for i in recs])

