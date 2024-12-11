# ui/app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from src.preprocessing.data_cleaner import DataCleaner
from src.models.svm_classifier import ViewerClassifier
from src.recommender.recommendation_engine import RecommendationEngine
from tqdm import tqdm

# Initialize Flask application
app = Flask(__name__)

# Initialize components
data_cleaner = DataCleaner()
classifier = ViewerClassifier()
recommender = RecommendationEngine()

# Load and prepare data
def load_data():
    movies_path = 'data/raw/movies.csv'
    viewing_path = 'data/raw/user_viewing.csv'
    
    if not (os.path.exists(movies_path) and os.path.exists(viewing_path)):
        print("Generating synthetic data...")
        from data.generate_movie_data import generate_movie_dataset, generate_user_viewing_data
        movies_df = generate_movie_dataset(n_movies=10000)
        viewing_df = generate_user_viewing_data(movies_df, n_users=5000)
    else:
        print("Loading existing data...")
        movies_df = pd.read_csv(movies_path)
        viewing_df = pd.read_csv(viewing_path)
    
    return movies_df, viewing_df

# Load data at startup
movies_df, viewing_df = load_data()

# First clean the data
print("Cleaning viewing data...")
cleaned_viewing_df = data_cleaner.clean_dataset(viewing_df)

# Then extract features from cleaned data
print("Extracting features...")
user_features = data_cleaner.extract_features(cleaned_viewing_df)

print("Training classifier...")
classifier.train(user_features)

@app.route('/')
def index():
    # Get some basic stats for the frontend
    stats = {
        'total_movies': len(movies_df),
        'total_users': viewing_df['user_id'].nunique(),
        'avg_rating': movies_df['rating'].mean().round(2),
        'popular_genres': movies_df['genre'].value_counts().head(5).to_dict()
    }
    return render_template('index.html', stats=stats)

@app.route('/process_user', methods=['POST'])
def process_user():
    try:
        user_data = request.json
        print(f"Received user data: {user_data}")
        
        # Create features from user data
        user_df = pd.DataFrame([{
            'watch_time': user_data['watch_time'],
            'completion_rate': user_data['completion_rate'],
            'rating': user_data['rating']
        }])
        
        # Process the data
        cleaned_data = data_cleaner.clean_dataset(user_df)
        features = data_cleaner.extract_features(cleaned_data)
        segment = classifier.predict(features)[0]
        
        # Get recommendations using user's rating
        recommendations = recommender.get_recommendations(
            segment,
            movies_df,
            user_rating=user_data['rating'],
            n_recommendations=10
        )
        
        return jsonify({
            'segment': segment,
            'recommendations': recommendations
        })
    except Exception as e:
        import traceback
        print(f"Error processing request: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'An error occurred processing your request',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)