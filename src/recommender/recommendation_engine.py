import pandas as pd
from typing import List, Dict

# src/recommender/recommendation_engine.py

class RecommendationEngine:
    def __init__(self):
        self.segment_preferences = {
            'casual': {
                'max_duration': 120,  # 2 hours
                'preferred_genres': ['Comedy', 'Romance', 'Family'],
                'min_rating': 3.5,
                'rating_window': 1.0  # Will look for movies within +/- 1.0 of user's rating
            },
            'regular': {
                'max_duration': 150,  # 2.5 hours
                'preferred_genres': ['Action', 'Drama', 'Thriller'],
                'min_rating': 3.0,
                'rating_window': 1.5
            },
            'enthusiast': {
                'max_duration': 180,  # 3 hours
                'preferred_genres': ['Sci-Fi', 'Mystery', 'Documentary'],
                'min_rating': 2.5,
                'rating_window': 2.0
            }
        }
    
    def get_recommendations(self, 
                          user_segment: str, 
                          movie_data: pd.DataFrame,
                          user_rating: float = 3.0,
                          n_recommendations: int = 5) -> List[Dict]:
        """
        Generate personalized recommendations based on user segment and rating
        """
        preferences = self.segment_preferences[user_segment]
        rating_window = preferences['rating_window']
        
        # Calculate rating range based on user's rating
        min_acceptable_rating = max(1.0, user_rating - rating_window)
        max_acceptable_rating = min(5.0, user_rating + rating_window)
        
        # Filter movies based on segment preferences and rating range
        filtered_movies = movie_data[
            (movie_data['duration'] <= preferences['max_duration']) &
            (movie_data['rating'] >= min_acceptable_rating) &
            (movie_data['rating'] <= max_acceptable_rating) &
            (movie_data['genre'].isin(preferences['preferred_genres']))
        ]
        
        # If no movies match the criteria, broaden the rating range
        if len(filtered_movies) < n_recommendations:
            filtered_movies = movie_data[
                (movie_data['duration'] <= preferences['max_duration']) &
                (movie_data['rating'] >= preferences['min_rating']) &
                (movie_data['genre'].isin(preferences['preferred_genres']))
            ]
        
        # If still no movies, return any top-rated movies
        if len(filtered_movies) == 0:
            filtered_movies = movie_data.nlargest(n_recommendations, 'rating')
        
        # Sort by a combination of rating similarity to user rating and popularity
        filtered_movies['rating_diff'] = abs(filtered_movies['rating'] - user_rating)
        filtered_movies['score'] = (
            filtered_movies['rating'] * 0.4 + 
            (5 - filtered_movies['rating_diff']) * 0.6
        )
        
        # Get diverse recommendations
        recommended_movies = filtered_movies.nlargest(n_recommendations, 'score')
        
        # Convert to list of dictionaries with selected fields
        recommendations = recommended_movies[[
            'title', 'genre', 'rating', 'duration'
        ]].to_dict('records')
        
        return recommendations