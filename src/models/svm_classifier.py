# src/models/svm_classifier.py

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, List

class ViewerClassifier:
    def __init__(self):
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.segments = ['casual', 'regular', 'enthusiast']
        
    def train(self, features: np.ndarray) -> None:
        """
        Train the SVM classifier using viewer features
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create initial labels using basic thresholds
        labels = self._create_initial_labels(scaled_features)
        
        # Train SVM
        self.svm.fit(scaled_features, labels)
    
    def predict(self, features: np.ndarray) -> List[str]:
        """
        Predict viewer segment for given features
        """
        scaled_features = self.scaler.transform(features)
        predictions = self.svm.predict(scaled_features)
        return [self.segments[p] for p in predictions]
    
    def _create_initial_labels(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Create initial labels based on engagement scores
        """
        engagement_scores = scaled_features[:, 1]  # Use engagement score feature
        labels = np.zeros(len(engagement_scores), dtype=int)
        
        # Set thresholds for segmentation
        lower_threshold = np.percentile(engagement_scores, 33)
        upper_threshold = np.percentile(engagement_scores, 66)
        
        labels[(engagement_scores >= lower_threshold) & 
               (engagement_scores < upper_threshold)] = 1
        labels[engagement_scores >= upper_threshold] = 2
        
        return labels