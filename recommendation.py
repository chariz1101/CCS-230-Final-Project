import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
def load_data():
    movies = pd.read_csv('movies.dat', sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    ratings = pd.read_csv('ratings.dat', sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    users = pd.read_csv('users.dat', sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')
    return movies, ratings, users

# Occupation mapping
occupation_map = {
    0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}

# TF-IDF and cosine similarity
def prepare_model(movies):
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    movie_indices = pd.Series(movies.index, index=movies['MovieID'])
    return cosine_sim, movie_indices

def recommend_movies(genre_list=None, user_id=None, title=None, top_n=10, min_ratings=50):
    global movies, ratings, users, cosine_sim, movie_indices, avg_ratings

    if 'avg_ratings' not in globals():
        movie_ratings = pd.merge(movies, ratings, on='MovieID')
        avg_ratings = movie_ratings.groupby(['MovieID', 'Title', 'Genres']).agg({'Rating': ['mean', 'count']})
        avg_ratings.columns = ['AverageRating', 'RatingCount']
        avg_ratings = avg_ratings.reset_index()

    recommended = pd.DataFrame()

    if title:
        title_row = movies[movies['Title'].str.contains(title, case=False, na=False)]
        if not title_row.empty:
            movie_id = title_row.iloc[0]['MovieID']
            idx = movie_indices[movie_id]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            similar_indices = [i[0] for i in sim_scores[1:top_n+1]]
            recommended = movies.iloc[similar_indices][['Title', 'Genres']]

    elif genre_list:
        genre_query = "|".join(genre_list)
        genre_filtered = avg_ratings[avg_ratings['Genres'].str.contains(genre_query, case=False)]
        genre_filtered = genre_filtered[genre_filtered['RatingCount'] >= min_ratings]

        if genre_filtered.empty:
            return pd.DataFrame({'Title': [], 'Genres': []})

        if user_id and user_id in users['UserID'].values:
            user_data = users[users['UserID'] == user_id].iloc[0]
            similar_users = users[(users['Occupation'] == user_data['Occupation']) & (users['Gender'] == user_data['Gender'])]
            sim_user_ratings = ratings[ratings['UserID'].isin(similar_users['UserID'])]
            sim_avg_ratings = sim_user_ratings.groupby('MovieID').agg({'Rating': 'mean'}).reset_index()
            genre_filtered = pd.merge(genre_filtered, sim_avg_ratings, on='MovieID', how='left')
            genre_filtered['AdjustedRating'] = (genre_filtered['AverageRating'] + genre_filtered['Rating'].fillna(0)) / 2
            genre_filtered = genre_filtered.sort_values(by='AdjustedRating', ascending=False)
        else:
            genre_filtered = genre_filtered.sort_values(by='AverageRating', ascending=False)

        top_movie_ids = genre_filtered['MovieID'].head(top_n).values
        similar_movies = set()
        for movie_id in top_movie_ids:
            idx = movie_indices[movie_id]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            similar_indices = [i[0] for i in sim_scores[1:6]]
            similar_movies.update(movies.iloc[similar_indices]['MovieID'].values)

        final_selection = movies[movies['MovieID'].isin(list(top_movie_ids) + list(similar_movies))]
        recommended = final_selection[['Title', 'Genres']].drop_duplicates().head(top_n)

    return recommended

# Load and prepare on import
movies, ratings, users = load_data()
cosine_sim, movie_indices = prepare_model(movies)
