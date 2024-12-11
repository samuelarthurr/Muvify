# data/generate_movie_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def generate_movie_dataset(n_movies=10000, output_path='data/raw/movies.csv'):
    """Generate a synthetic movie dataset"""
    print("Generating movie data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define possible genres and their weights
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance',
              'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Generate basic movie data
    data = {
        'movie_id': range(1, n_movies + 1),
        'title': [f"Movie_{i}" for i in range(1, n_movies + 1)],
        'release_year': np.random.randint(1990, 2024, n_movies),
        'duration': np.random.normal(120, 30, n_movies).astype(int).clip(60, 240),
        'genre': np.random.choice(genres, n_movies),
        'rating': np.random.normal(3.5, 0.8, n_movies).clip(1, 5).round(1),
        'popularity_score': np.random.exponential(50, n_movies).clip(1, 100).round(1),
        'budget': np.random.exponential(50000000, n_movies).astype(int)
    }
    
    # Calculate revenue based on budget with some randomness
    roi_multiplier = np.random.normal(2.5, 1.0, n_movies).clip(0.5, 5)
    data['revenue'] = (data['budget'] * roi_multiplier).astype(int)
    
    # Create DataFrame
    movies_df = pd.DataFrame(data)
    
    # Save to CSV
    movies_df.to_csv(output_path, index=False)
    print(f"Generated {n_movies} movies")
    return movies_df

def generate_user_viewing_data(movies_df, n_users=5000, output_path='data/raw/user_viewing.csv'):
    """Generate synthetic user viewing data"""
    print("Generating user viewing data...")
    viewing_records = []
    
    for user_id in tqdm(range(1, n_users + 1)):
        # Each user watches between 5 and 50 movies
        n_movies_watched = np.random.randint(5, 51)
        
        # Select random movies for this user
        watched_movies = movies_df.sample(n=n_movies_watched)
        
        for _, movie in watched_movies.iterrows():
            # Generate viewing data
            completion_rate = np.random.beta(7, 3)  # Most users complete movies
            watch_time = int(movie['duration'] * completion_rate)
            
            # Generate rating with some correlation to movie's average rating
            base_rating = movie['rating']
            user_rating = min(5, max(1, np.random.normal(base_rating, 0.5)))
            
            viewing_records.append({
                'user_id': user_id,
                'movie_id': movie['movie_id'],
                'watch_time': watch_time,
                'completion_rate': completion_rate,
                'rating': round(user_rating, 1),
                'date_watched': (datetime(2024, 1, 1) - 
                               timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            })
    
    # Create DataFrame and save to CSV
    viewing_df = pd.DataFrame(viewing_records)
    viewing_df.to_csv(output_path, index=False)
    print(f"Generated {n_users} users with {len(viewing_df)} viewing records")
    return viewing_df

if __name__ == '__main__':
    # Generate datasets
    movies_df = generate_movie_dataset()
    viewing_df = generate_user_viewing_data(movies_df)