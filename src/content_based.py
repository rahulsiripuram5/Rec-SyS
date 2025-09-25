import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_content_similarity_matrix(path="data/ml-100k/u.item"):
    """Builds a cosine similarity matrix based on movie genres using TF-IDF."""
    # Define column names for u.item
    columns = [
        'item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
        'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 
        'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 
        'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'
    ]
    movies_df = pd.read_csv(path, sep='|', names=columns, encoding='latin-1')
    
    # Create a string of genres for each movie
    genre_cols = columns[5:]
    movies_df['genres'] = movies_df[genre_cols].apply(
        lambda row: ' '.join(row.index[row.astype(bool)]), axis=1
    )
    
    # Use TF-IDF to vectorize the genre strings
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Compute the cosine similarity matrix
    print("Building content-based similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping from item_id to dataframe index
    movie_id_to_index = pd.Series(movies_df.index, index=movies_df['item_id']).to_dict()
    
    return cosine_sim, movies_df, movie_id_to_index

# This block allows you to run this file as a script for a quick demo
if __name__ == '__main__':
    from utils import load_movie_titles
    
    similarity_matrix, movies, id_to_idx = build_content_similarity_matrix()
    movie_titles = load_movie_titles()
    
    # Get recommendations for 'Toy Story (1995)' (item_id=1)
    movie_id = 1
    movie_idx = id_to_idx[movie_id]
    
    # Get pairwise similarity scores for the movie
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get scores of the 5 most similar movies (index 0 is the movie itself)
    sim_scores = sim_scores[1:6]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Get movie titles
    similar_movies = [movies.iloc[i]['title'] for i in movie_indices]
    
    print(f"Movies similar to '{movie_titles[movie_id]}':")
    for movie in similar_movies:
        print(f"- {movie}")