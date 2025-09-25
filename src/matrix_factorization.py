import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def get_ratings_data(path="data/ml-100k/u.data"):
    """Loads and prepares the MovieLens ratings data for Surprise."""
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    return df, data

def train_svd_model(data):
    """Trains the SVD model on the full dataset for prediction and reports RMSE on a test set."""
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    
    print("Training SVD model...")
    algo.fit(trainset)
    
    print("Evaluating on test set...")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    
    return algo

# This block allows you to run this file as a script for a quick demo
if __name__ == '__main__':
    from utils import load_movie_titles
    
    ratings_df, data = get_ratings_data()
    svd_algo = train_svd_model(data)
    movie_titles = load_movie_titles()
    
    user_id = 405
    # Get a list of all movie ids
    all_movie_ids = ratings_df['item_id'].unique()
    # Get a list of movies the user has already rated
    rated_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['item_id']
    # Get a list of movies the user has not rated
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids.values]

    # Predict ratings for unrated movies
    predictions = [svd_algo.predict(user_id, movie_id) for movie_id in unrated_movie_ids]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    print(f"\nTop 5 recommendations for user {user_id} (SVD):")
    for pred in predictions[:5]:
        print(f"- {movie_titles[pred.iid]} (Predicted rating: {pred.est:.2f})")