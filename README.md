# Muvify
 RSBP_FP

| Group Member            | Student ID   | Class        |
|-------------------------|--------------|--------------|
| Samuel Arthur Gamalliel | 5025221109   | RSBP (IUP)   |
| Surya Prima Pradana     | 5025221076   | RSBP (IUP)   |
| Ralfazza Rajariandhana  | 5025221081   | RSBP (IUP)   |



# Movie Recommendation System

A personalized movie recommendation system that categorizes users based on their viewing preferences, behavior, and ratings data. The system uses Support Vector Machine (SVM) classification to segment viewers into three categories: casual, regular, and enthusiast viewers.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Data Structure](#data-structure)
- [API Documentation](#api-documentation)

## Project Structure
```
movie_recommender/
├── data/
│   ├── raw/                  # Raw dataset files
│   │   ├── movies.csv        # Generated movie dataset
│   │   └── user_viewing.csv  # Generated user viewing data
│   └── processed/            # Processed dataset files
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_cleaner.py   # Data cleaning and feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   └── svm_classifier.py # SVM model for user segmentation
│   ├── recommender/
│   │   ├── __init__.py
│   │   └── recommendation_engine.py # Recommendation logic
│   └── utils/
│       ├── __init__.py
│       └── metrics.py        # Evaluation metrics
├── ui/
│   ├── static/
│   │   └── styles.css       # UI styling
│   ├── templates/
│   │   ├── index.html       # Main page template
│   │   └── recommendations.html # Recommendations template
│   └── app.py              # Flask application
├── setup.py               # Project setup configuration
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features
- User segmentation into three categories:
  - Casual viewers
  - Regular viewers
  - Enthusiast viewers
- Personalized movie recommendations based on:
  - Viewing history
  - Watch time patterns
  - Completion rates
  - Rating preferences
- Web-based user interface
- Real-time recommendation generation
- Synthetic data generation for testing

## Technologies Used
- Python 3.12+
- Flask (Web Framework)
- pandas (Data Processing)
- scikit-learn (Machine Learning)
- NumPy (Numerical Computing)
- HTML/CSS (Frontend)

## Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
.\venv\Scripts\activate   # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage
1. Start the Flask application:
```bash
python ui/app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Enter your viewing preferences:
   - Average watch time
   - Completion rate
   - Rating preferences

## Algorithms

### 1. Data Preprocessing
- Data cleaning
  - Remove incomplete views (CompletionRate < 20%)
  - Remove invalid ratings (Rating = 0)
- Feature engineering
  - Engagement score calculation
  - Watch time normalization
  - Completion rate processing

### 2. User Segmentation (SVM)
- Support Vector Machine Classification
  - Kernel: RBF (Radial Basis Function)
  - Features used:
    - Normalized watch time
    - Engagement score
    - Completion rate
  - Output classes:
    - Casual viewer
    - Regular viewer
    - Enthusiast viewer

### 3. Recommendation Engine
- Filtering criteria:
  - Duration preferences per segment
  - Genre preferences per segment
  - Rating window based on user input
- Scoring system:
  - Rating similarity weight: 60%
  - Overall rating weight: 40%
- Fallback mechanisms:
  - Rating window expansion
  - Genre expansion
  - Minimum rating threshold

## Data Structure

### Movies Dataset (movies.csv)
- movie_id (int): Unique identifier
- title (string): Movie title
- release_year (int): Year of release
- duration (int): Duration in minutes
- genre (string): Movie genre
- rating (float): Average rating (1-5)
- popularity_score (float): Popularity metric
- budget (int): Production budget
- revenue (int): Box office revenue

### User Viewing Data (user_viewing.csv)
- user_id (int): Unique identifier
- movie_id (int): Reference to movies dataset
- watch_time (int): Minutes watched
- completion_rate (float): Percentage watched
- rating (float): User rating (1-5)
- date_watched (datetime): Viewing timestamp

## API Documentation

### Endpoints

#### GET /
- Returns the main page with the user input form

#### POST /process_user
- Input:
```json
{
  "watch_time": float,
  "completion_rate": float,
  "rating": float
}
```
- Output:
```json
{
  "segment": string,
  "recommendations": [
    {
      "title": string,
      "genre": string,
      "rating": float,
      "duration": int
    }
  ]
}
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
