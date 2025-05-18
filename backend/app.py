from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = "a9d902d7338397e2deda22858c168821"

def get_poster_url(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {'api_key': TMDB_API_KEY, 'query': movie_title}
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Load and preprocess data once
movies = pd.read_csv('movies.dat', sep='::', engine='python',
                     names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
ratings = pd.read_csv('ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
users = pd.read_csv('users.dat', sep='::', engine='python',
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')

movie_ratings = pd.merge(movies, ratings, on='MovieID')
avg_ratings = movie_ratings.groupby(['MovieID', 'Title', 'Genres']).agg({'Rating': ['mean', 'count']})
avg_ratings.columns = ['AverageRating', 'RatingCount']
avg_ratings = avg_ratings.reset_index()

tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies['MovieID'])

@app.route('/app/recommend', methods=['POST'])
def recommend():
    data = request.json
    genre_list = data.get('genres', [])
    user_id = int(data.get('userId', 0)) if data.get('userId', '').isdigit() else None

    recommended = pd.DataFrame()

    if genre_list:
        genre_query = "|".join(genre_list)
        genre_filtered = avg_ratings[avg_ratings['Genres'].str.contains(genre_query, case=False)]
        genre_filtered = genre_filtered[genre_filtered['RatingCount'] >= 50]

        if user_id and user_id in users['UserID'].values:
            user_data = users[users['UserID'] == user_id].iloc[0]
            similar_users = users[
                (users['Occupation'] == user_data['Occupation']) &
                (users['Gender'] == user_data['Gender'])
            ]
            sim_user_ratings = ratings[ratings['UserID'].isin(similar_users['UserID'])]
            sim_avg_ratings = sim_user_ratings.groupby('MovieID').agg({'Rating': 'mean'}).reset_index()
            genre_filtered = pd.merge(genre_filtered, sim_avg_ratings, on='MovieID', how='left')
            genre_filtered['AdjustedRating'] = (genre_filtered['AverageRating'] + genre_filtered['Rating'].fillna(0)) / 2
            genre_filtered = genre_filtered.sort_values(by='AdjustedRating', ascending=False)
        else:
            genre_filtered = genre_filtered.sort_values(by='AverageRating', ascending=False)

        top_movie_ids = genre_filtered['MovieID'].head(10).values
        similar_movies = set()
        for movie_id in top_movie_ids:
            idx = movie_indices[movie_id]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            similar_indices = [i[0] for i in sim_scores[1:6]]
            similar_movies.update(movies.iloc[similar_indices]['MovieID'].values)

        final_selection = movies[movies['MovieID'].isin(list(top_movie_ids) + list(similar_movies))]
        recommended = final_selection[['Title', 'Genres']].drop_duplicates().head(10)

    result = []
    for _, row in recommended.iterrows():
        result.append({
            "title": row['Title'],
            "genres": row['Genres'],
            "poster": get_poster_url(row['Title'])
        })

    return jsonify(result)
