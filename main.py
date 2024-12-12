import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from dotenv import load_dotenv
import os
import requests

app = Flask(__name__)

# Load the movie and rating datasets
movies = pd.read_csv(os.path.join('DataSet', 'movies.csv'))
ratings = pd.read_csv(os.path.join('DataSet', 'ratings.csv'))

# TMDb API key
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Prepare content-based recommendations using genres and titles
movies['genres'] = movies['genres'].fillna('')  # Handle missing genres
movies['title'] = movies['title'].fillna('')    # Handle missing titles
tfidf_genres = TfidfVectorizer(stop_words='english')  # Initialize TF-IDF Vectorizer for genres
tfidf_titles = TfidfVectorizer(stop_words='english')   # Initialize TF-IDF Vectorizer for titles

tfidf_matrix_genres = tfidf_genres.fit_transform(movies['genres'])  # Convert 'genres' into a matrix of TF-IDF features
tfidf_matrix_titles = tfidf_titles.fit_transform(movies['title'])    # Convert 'title' into a matrix of TF-IDF features

# Calculate cosine similarity matrix for the movies based on genres and titles
content_similarity_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
content_similarity_titles = cosine_similarity(tfidf_matrix_titles, tfidf_matrix_titles)

# Combine genre and title similarities
content_similarity = (content_similarity_genres + content_similarity_titles) / 2

# Function to validate user ID and movie title inputs
def validate_user_id(user_id):
    return isinstance(user_id, int) and user_id > 0

def validate_movie_title(movie_title):
    return isinstance(movie_title, str) and len(movie_title) > 0

# Function to get movie poster from TMDb
def get_movie_poster(movie_title):
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title
    }
    try:
        response = requests.get(search_url, params=params)
        data = response.json()

        if 'results' in data and data['results']:
            # Use the best-matched movie from TMDb
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                # Return complete poster URL
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        # If no poster found, log the issue and return placeholder
        print(f"No poster found for {movie_title}, using placeholder.")
        return '/static/images/placeholder.png'  # Ensure you have a placeholder image path here
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error fetching poster for {movie_title}: {str(e)}")
        return '/static/images/placeholder.png'

def get_top_rated_movies_by_genre(num_movies=10):
    # Merge movies and ratings
    movies_ratings = pd.merge(movies, ratings, on='movieId')
    # Calculate average rating for each movie
    movies_ratings_avg = movies_ratings.groupby('movieId').agg({
        'rating': 'mean',
        'title': 'first',
        'genres': 'first'
    }).reset_index()

    movies_ratings_avg = movies_ratings_avg.sort_values(by='rating', ascending=False)
    genre_top_movies = {}

    # For each genre, find the top n movies
    for genre in set('|'.join(movies_ratings_avg['genres']).split('|')):
        genre_movies = movies_ratings_avg[movies_ratings_avg['genres'].str.contains(genre)]
        top_genre_movies = genre_movies.head(num_movies)[['title', 'rating']]
        
        # Add movie poster URLs
        top_genre_movies['poster'] = top_genre_movies['title'].apply(get_movie_poster)
        
        genre_top_movies[genre] = top_genre_movies.to_dict(orient='records')
    
    return genre_top_movies


# Function to get movie recommendations based on genre and title similarity (content-based filtering)
def get_content_based_recommendations(movie_title, num_recommendations=15):
    movie_title = movie_title.strip().lower()
    titles = movies['title'].tolist()
    best_match, score = process.extractOne(movie_title, titles)

    if score < 80:  # Adjust the threshold for matching
        return [], None  # Return empty if no good match is found

    idx = movies[movies['title'] == best_match].index[0]
    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter recommendations and ensure they are not the same as the searched movie
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    recommended_titles = movies['title'].iloc[movie_indices].tolist()
    recommended_posters = [get_movie_poster(title) for title in recommended_titles]
    
    return [(title, poster or '/static/images/placeholder.png') for title, poster in zip(recommended_titles, recommended_posters)], best_match


# Create a user-item matrix for user-based collaborative filtering
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to recommend movies for a given user based on similar users (user-based filtering)
def recommend_movies_for_user(user_id, num_recommendations=5):
    try:
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        top_user = similar_users.index[0]
        user_rated_movies = user_item_matrix.loc[user_id]
        similar_user_rated_movies = user_item_matrix.loc[top_user]
        
        recommendations = similar_user_rated_movies[(similar_user_rated_movies > 0) & (user_rated_movies == 0)]
        movie_ids = recommendations.sort_values(ascending=False).head(num_recommendations).index.tolist()
        recommended_titles = movies[movies['movieId'].isin(movie_ids)]['title'].tolist()
        recommended_posters = [get_movie_poster(title) for title in recommended_titles]
        return [(title, poster) for title, poster in zip(recommended_titles, recommended_posters)]
    except KeyError:
        return []  # Return empty if the user ID does not exist

# Flask route for the home page
@app.route('/')
def index():
    return render_template('saki.html')

# Flask route to handle recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id')
    movie_title = request.form.get('movie_title')
    
    # Validate user ID
    if user_id:
        if not validate_user_id(int(user_id)):
            return jsonify({'error': 'Invalid User ID. It should be a positive integer.'}), 400

    # Validate movie title
    if movie_title and not validate_movie_title(movie_title):
        return jsonify({'error': 'Invalid Movie Title. It should be a non-empty string.'}), 400
    
    # Get recommendations
    searched_movie_details = {}
    if user_id:
        recommendations = recommend_movies_for_user(int(user_id))
    else:
        recommendations, best_match = get_content_based_recommendations(movie_title)
        searched_movie_details['title'] = best_match
        searched_movie_details['poster'] = get_movie_poster(best_match)

    return jsonify({
        'searched_movie': searched_movie_details,
        'recommendations': recommendations
    })

# Flask route for autocomplete feature
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').strip().lower()
    titles = movies['title'].str.lower().tolist()

    # Filter titles that match the query
    matches = [title for title in titles if query in title][:10]  # Limit to 10 results
    return jsonify(matches)

# Route to get top-rated movies by genre
@app.route('/top_movies_by_genre', methods=['GET'])
def top_movies_by_genre():
    top_movies_by_genre = get_top_rated_movies_by_genre()
    return jsonify(top_movies_by_genre)

if __name__ == '__main__':
    app.run(debug=True)
