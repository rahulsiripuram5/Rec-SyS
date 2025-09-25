# In src/hybrid_recommender.py

def generate_hybrid_recommendations(user_id, svd_algo, content_sim_matrix, movies_df, movie_id_to_index, all_movie_ids, user_train_ids, alpha=0.5, top_k=10):
    """
    Generates hybrid recommendations, using the user's 'training' items to build a profile
    and excluding them from the final recommendation list.
    """
    # Candidate movies are all movies EXCEPT the ones in the user's training profile.
    candidate_movie_ids = [mid for mid in all_movie_ids if mid not in user_train_ids]
    
    # 1. SVD Scores
    svd_preds = [svd_algo.predict(user_id, movie_id) for movie_id in candidate_movie_ids]
    max_svd_rating = 5.0
    svd_scores = {pred.iid: pred.est / max_svd_rating for pred in svd_preds}
    
    # 2. Content-Based Scores (built ONLY from the user's training profile)
    content_scores = {}
    rated_movies_indices = [movie_id_to_index[mid] for mid in user_train_ids if mid in movie_id_to_index]
    
    if rated_movies_indices:
        avg_sim_scores = content_sim_matrix[rated_movies_indices, :].mean(axis=0)
        
        for movie_idx, score in enumerate(avg_sim_scores):
            movie_id = movies_df.iloc[movie_idx]['item_id']
            if movie_id in candidate_movie_ids:
                content_scores[movie_id] = score
    
    # 3. Combine Scores
    final_scores = {}
    for movie_id in candidate_movie_ids:
        svd_score = svd_scores.get(movie_id, 0)
        content_score = content_scores.get(movie_id, 0)
        final_scores[movie_id] = (alpha * svd_score) + ((1 - alpha) * content_score)
        
    sorted_recommendations = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_recommendations[:top_k]

# This block allows you to run this file as a script for a quick demo
if __name__ == '__main__':
    from matrix_factorization import get_ratings_data, train_svd_model
    from content_based import build_content_similarity_matrix
    from utils import load_movie_titles
    
    # Train models to get necessary components
    print("--- Preparing models for hybrid recommendation demo ---")
    ratings, data = get_ratings_data()
    svd = train_svd_model(data)
    sim_matrix, movies, id_to_idx = build_content_similarity_matrix()
    titles = load_movie_titles()
    
    # Generate recommendations for a user
    user = 405
    
    # --- FIX IS HERE ---
    # We need to define all_movie_ids and the user's rated items (user_train_ids)
    all_movie_ids = ratings['item_id'].unique()
    user_train_ids = ratings[ratings['user_id'] == user]['item_id'].values

    recommendations = generate_hybrid_recommendations(
        user_id=user,
        svd_algo=svd,
        content_sim_matrix=sim_matrix,
        movies_df=movies,
        movie_id_to_index=id_to_idx,
        all_movie_ids=all_movie_ids,      # <-- ADDED
        user_train_ids=user_train_ids,    # <-- ADDED
        alpha=0.6,
        top_k=5
    )
    
    print(f"\nTop 5 hybrid recommendations for user {user}:")
    for movie_id, score in recommendations:
        print(f"- {titles[movie_id]} (Hybrid Score: {score:.3f})")
