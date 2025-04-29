# 📚 Imports
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# 📚 Load saved models
with open('model/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('model/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('model/movie_indices.pkl', 'rb') as f:
    movie_indices = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

movies = pd.read_csv('model/movies_metadata.csv')

# 📚 Initialize app
app = Flask(__name__)

# 📚 Recommend function
def recommend_movies(movie_features_text=None, top_n=10, is_new_movie=False, existing_movie_id=None):
    if not is_new_movie:
        idx = movie_indices.get(existing_movie_id, None)
        if idx is None:
            raise ValueError("Movie ID not found!")
        distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
        sim_indices = indices.flatten()[1:]  # exclude self
        recommended_movies = movies.iloc[sim_indices][['movieId', 'title']]
    else:
        movie_vec = tfidf_vectorizer.transform([movie_features_text])
        distances, indices = knn_model.kneighbors(movie_vec, n_neighbors=top_n)
        sim_indices = indices.flatten()
        recommended_movies = movies.iloc[sim_indices][['movieId', 'title']]
    
    return recommended_movies.reset_index(drop=True)

# 📚 Routes
@app.route('/recommend_by_movieid', methods=['POST'])
def recommend_by_movieid():
    data = request.get_json()
    movie_id = data.get('movieId')
    if movie_id is None:
        return jsonify({'error': 'movieId not provided'}), 400
    try:
        recs = recommend_movies(existing_movie_id=movie_id)
        return jsonify(recs.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend_by_features', methods=['POST'])
def recommend_by_features():
    data = request.get_json()
    features_text = data.get('features')
    if features_text is None:
        return jsonify({'error': 'features not provided'}), 400
    try:
        recs = recommend_movies(movie_features_text=features_text, is_new_movie=True)
        return jsonify(recs.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 📚 Home
@app.route('/')
def home():
    return "✅ Movie Recommendation Backend Running"

# 📚 Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
