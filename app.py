from flask import Flask, request, jsonify, render_template
from recommendation import load_data, recommend_movies
from poster_fetcher import get_poster_url

app = Flask(__name__)

movies, ratings, users = load_data()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/rekom')
def rekom():
    return render_template('rekom.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genres = data.get('genres', [])
    user_id = data.get('user_id')
    title = data.get('title')

    uid = int(user_id) if user_id and str(user_id).isdigit() else None

    recommendations = recommend_movies(genre_list=genres, user_id=uid, title=title)
    rec_list = recommendations.to_dict(orient='records')

    for movie in rec_list:
        movie['poster'] = get_poster_url(movie['Title'])

    return jsonify(rec_list)

if __name__ == '__main__':
    app.run(debug=True)
