import pandas as pd

def load_movie_titles(path="data/ml-100k/u.item"):
    """
    Loads MovieLens movie titles into a dictionary {item_id: title}
    """
    # MovieLens u.item is '|' separated
    movies = pd.read_csv(path, sep='|', header=None, encoding='latin-1')
    movies = movies[[0, 1]]
    movies.columns = ['item_id', 'title']
    return dict(zip(movies['item_id'], movies['title']))
