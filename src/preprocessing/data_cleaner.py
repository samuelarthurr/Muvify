# src/preprocessing/data_cleaner.py

import pandas as pd
import numpy as np
from typing import Tuple

class DataCleaner:
    def __init__(self, min_completion_rate: float = 0.2):
        self.min_completion_rate = min_completion_rate
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing invalid entries and calculating engagement metrics
        """
        # Create a copy to avoid modifying the original DataFrame
        cleaned_df = df.copy()
        
        # Remove incomplete views and invalid ratings
        cleaned_df = cleaned_df[
            (cleaned_df['completion_rate'] >= self.min_completion_rate) &
            (cleaned_df['rating'] > 0)
        ]
        
        # Calculate engagement score
        cleaned_df['EngagementScore'] = self._calculate_engagement_score(cleaned_df)
        
        # Normalize watch time
        cleaned_df['NormalizedWatchTime'] = self._normalize_watch_time(cleaned_df)
        
        return cleaned_df
    
    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate engagement score based on rating and completion rate
        """
        return (df['rating'] * 0.7 + df['completion_rate'] * 0.3) / 2
    
    def _normalize_watch_time(self, df: pd.DataFrame) -> pd.Series:
        """
        Normalize watch time across different movie lengths
        """
        if len(df) == 1:  # If only one entry
            return pd.Series([0], index=df.index)  # Return 0 as the normalized value
        return (df['watch_time'] - df['watch_time'].mean()) / (df['watch_time'].std() or 1)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for SVM classification
        Must be called after clean_dataset
        """
        # First clean the dataset if not already cleaned
        if 'NormalizedWatchTime' not in df.columns:
            df = self.clean_dataset(df)
        
        features = np.column_stack((
            df['NormalizedWatchTime'],
            df['EngagementScore'],
            df['completion_rate']
        ))
        return features