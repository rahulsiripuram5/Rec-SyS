# In src/evaluate_hybrid.py

import numpy as np
from tqdm import tqdm

from matrix_factorization import get_ratings_data, train_svd_model
from content_based import build_content_similarity_matrix
from hybrid_recommender import generate_hybrid_recommendations

K = 10

def precision_recall_at_k(recommendations, hold_out_items):
    """Calculates precision and recall for a list of recommendations."""
    recommended_ids = {item_id for item_id, score in recommendations}
    hits = len(recommended_ids.intersection(hold_out_items))
    
    precision = hits / K
    recall = hits / len(hold_out_items) if hold_out_items else 0
    
    return precision, recall

def main():
    """Main function to run the full training and evaluation pipeline."""
    print("--- Loading data and training models ---")
    ratings_df, data = get_ratings_data()
    svd_algo = train_svd_model(data)
    sim_matrix, movies_df, id_to_idx = build_content_similarity_matrix()
    all_movie_ids = ratings_df['item_id'].unique()
    print("\n--- Models trained successfully ---")
    
    all_users = ratings_df['user_id'].unique()
    precisions = []
    recalls = []
    
    print(f"\n--- Evaluating hybrid model for {len(all_users)} users (K={K}) ---")
    for user_id in tqdm(all_users, desc="Evaluating Users"):
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        # Ground truth (test set): movies the user liked (rating >= 4).
        hold_out_items = set(user_ratings[user_ratings['rating'] >= 4]['item_id'])
        
        if not hold_out_items:
            continue
            
        # Training set for this user: all items they rated EXCEPT the hold-out items.
        # This is what the model is allowed to "know" about the user.
        user_train_ids = set(user_ratings['item_id']) - hold_out_items
        
        # We need at least one item in the training profile to make content-based recommendations
        if not user_train_ids:
            continue
            
        recommendations = generate_hybrid_recommendations(
            user_id=user_id,
            svd_algo=svd_algo,
            content_sim_matrix=sim_matrix,
            movies_df=movies_df,
            movie_id_to_index=id_to_idx,
            all_movie_ids=all_movie_ids,
            user_train_ids=user_train_ids, # Pass the user's training items here
            alpha=0.6,
            top_k=K
        )
        
        p, r = precision_recall_at_k(recommendations, hold_out_items)
        precisions.append(p)
        recalls.append(r)
        
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    print("\n--- Overall Hybrid Model Performance ---")
    print(f"Average Precision@{K}: {avg_precision:.4f}")
    print(f"Average Recall@{K}:    {avg_recall:.4f}")
    print(f"Evaluated on {len(precisions)} users with relevant items.")

if __name__ == '__main__':
    main()