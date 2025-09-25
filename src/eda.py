import pandas as pd
import matplotlib.pyplot as plt

p = "data/ml-100k/u.data"
df = pd.read_csv(p, sep='\t', names=['user_id','item_id','rating','timestamp'])
print("Rows:", len(df))
print("Unique users:", df['user_id'].nunique())
print("Unique items:", df['item_id'].nunique())
print("Rating distribution:\n", df['rating'].value_counts().sort_index())
print(df.head())




n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
n_ratings = len(df)

sparsity = 1 - (n_ratings / (n_users * n_items))

print(f"Users: {n_users}, Items: {n_items}, Ratings: {n_ratings}")
print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")

# Top 5 users with most ratings
top_users = df['user_id'].value_counts().head()
print("\nTop 5 active users:\n", top_users)

# Top 5 movies with most ratings
top_items = df['item_id'].value_counts().head()
print("\nTop 5 popular movies:\n", top_items)



df['rating'].hist(bins=5)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.title("Ratings Distribution")
plt.savefig("images/ratings_distribution.png") # Save to a dedicated folder
plt.show()


