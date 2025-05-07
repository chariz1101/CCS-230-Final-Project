import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import requests

TMDB_API_KEY = "a9d902d7338397e2deda22858c168821"

def get_poster_url(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.dat', sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    ratings = pd.read_csv('ratings.dat', sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    users = pd.read_csv('users.dat', sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')
    return movies, ratings, users

# Map occupation codes to labels
occupation_map = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

movies, ratings, users = load_data()

# Preprocess
movie_ratings = pd.merge(movies, ratings, on='MovieID')
avg_ratings = movie_ratings.groupby(['MovieID', 'Title', 'Genres']).agg({'Rating': ['mean', 'count']})
avg_ratings.columns = ['AverageRating', 'RatingCount']
avg_ratings = avg_ratings.reset_index()

# TF-IDF
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies['MovieID'])

def recommend_movies(genre_list, user_id=None, top_n=10, min_ratings=50):
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
    return final_selection[['Title', 'Genres']].drop_duplicates().head(top_n)

# Streamlit App Pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Methodology", "Demo"])

if page == "Introduction":
    st.title("🎬 Movie Recommendation System")
    st.write("""
    Welcome to the Movie Recommendation System! This application uses AI-powered methods to help you discover movies 
    that match your tastes, based on genre preferences and user behavior. 

    **Purpose:** Reduce time searching for content by offering intelligent recommendations.
    
    **Functionality:**
    - Filter movies by genre
    - Adjust results using user profiles (gender, occupation)
    - Recommend similar movies based on content

    **Objectives:**
    - Enhance user experience
    - Provide smart, data-driven movie suggestions
    - Demonstrate machine learning techniques in recommender systems
    """)

elif page == "Methodology":
    st.title("🔍 Methodology")
    st.markdown("""
    This app combines three major approaches in recommender systems:

    **1. Content-Based Filtering**
    - Uses TF-IDF on genres to compute similarity between movies.

    **2. Collaborative Filtering (Lite)**
    - Adjusts recommendations based on similar users' preferences (by gender and occupation).

    **3. Popularity-Based Filtering**
    - Filters out movies with very few ratings.

    **Recommendation Logic:**
    - User selects genres.
    - (Optional) User provides their ID.
    - System recommends top-rated and similar movies.
    """)

elif page == "Demo":
    st.title("🎥 Try the Movie Recommender")

    all_genres = sorted(list(set(g for genre in movies['Genres'] for g in genre.split('|'))))
    selected_genres = st.multiselect("Select Genre(s):", all_genres)
    user_id = st.text_input("Optional: Enter your User ID (from 1 to 6040)")

    if st.button("Recommend"):
        uid = int(user_id) if user_id.isdigit() else None

        if uid in users['UserID'].values:
            user_demo = users[users['UserID'] == uid].iloc[0]
            occ = occupation_map.get(user_demo['Occupation'], 'Unknown')
            st.info(f"User Demographics: Gender - {user_demo['Gender']}, Age - {user_demo['Age']}, Occupation - {occ}")

        recs = recommend_movies(selected_genres, user_id=uid)
        if recs.empty:
            st.warning("No recommendations found. Try different genres or lower the rating threshold.")
        else:
            st.success("Top Recommended Movies:")
            for _, row in recs.iterrows():
                poster_url = get_poster_url(row['Title'])
                cols = st.columns([1, 3])
                if poster_url:
                    cols[0].image(poster_url, use_container_width=True)
                cols[1].markdown(f"**{row['Title']}**\n\n*Genres:* {row['Genres']}")
