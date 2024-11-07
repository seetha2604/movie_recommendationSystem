from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load movie data with additional columns
movies = pd.read_csv('movies.csv')  # Ensure columns include 'title', 'rating', 'year', 'genres', 'description'

# Clean up columns by stripping unnecessary spaces and handling NaN values
movies['genres'] = movies['genres'].fillna('').str.strip()  # Fill NaN with empty string and strip spaces
movies['description'] = movies['description'].fillna('').str.strip()  # Fill NaN with empty string and strip spaces
movies['year'] = movies['year'].fillna(2000).astype(int)  # Replace NaN with 2000 and convert to int

# Vectorize genres for similarity matching (not needed if we just filter by genre)
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(movies['genres'])

# KNN model for genre-based similarity (not needed if we filter by genre)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(genre_matrix)

# Recommendation function based on genre
def get_recommendations_by_genre(input_genre):
    # Filter the movies that contain the input_genre in their genres column
    genre_filtered_movies = movies[movies['genres'].str.contains(input_genre, case=False, na=False)]
    
    # Sort the filtered movies by rating in descending order (highest rating first)
    genre_filtered_movies_sorted = genre_filtered_movies.sort_values(by='rating', ascending=False)
    
    # Return relevant details: title, year, genres, description, and rating
    return genre_filtered_movies_sorted[['title', 'year', 'genres', 'description', 'rating']]

@app.route('/')
def index():
    return render_template('index.html')  # Render the form for genre input

@app.route('/recommend', methods=['POST'])
def recommend():
    genre_input = request.form['genre']  # Get the genre input from the form
    recommendations = get_recommendations_by_genre(genre_input)  # Get all movies with the selected genre
    return render_template('recommendations.html', genre=genre_input, recommendations=recommendations)  # Render recommendations

if __name__ == '__main__':
    app.run(debug=True)
