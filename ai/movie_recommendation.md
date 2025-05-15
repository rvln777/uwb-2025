# Movie recommendation system

# Task
Create an applicaation which takes movie title as input and returns list of movies similar to it.

### **Step 1: Install Required Libraries**
1. Install the necessary Python libraries:

   ```bash
   pip install numpy pandas surprise scikit-learn
   ```

---

### **Step 2: Import Required Libraries**
1. Start your script by importing the required libraries:

   ```python
   import pandas as pd
   import numpy as np
   from surprise import SVD
   from surprise import Dataset
   from surprise import Reader
   from sklearn.metrics.pairwise import cosine_similarity
   from sklearn.feature_extraction.text import CountVectorizer
   from collections import defaultdict
   ```

---

### **Step 3: Download and Load MovieLens Dataset**
1. Download the MovieLens dataset from [MovieLens website](https://grouplens.org/datasets/movielens/) or use the `MovieLens 100k` dataset.

2. Read the dataset using `pandas`:

   ```python
   # Load movies data
   movies = pd.read_csv('movies.csv')  # File will include movieId, title, genres columns

   # Load ratings data
   ratings = pd.read_csv('ratings.csv')  # File will include userId, movieId, rating columns

   # Explore the data:
   print(movies.head())
   print(ratings.head())
   ```

---

### **Step 4: Data Preprocessing**
1. Since the MovieLens dataset may contain genres, ensure the genres column is normalized if necessary.
2. For simplicity, we'll focus on collaborative filtering (user-item matrix), so ensure the dataset has the required columns: `userId`, `movieId`, and `rating`.

   ```python
   # Check for missing values and drop them if any
   print(ratings.isnull().sum())
   ratings = ratings.dropna()

   # Verify the data structure
   print(ratings.head())
   ```

---

### **Step 5: Create a User-Item Rating Matrix**
1. The `surprise` library handles user-item matrices internally, but let's construct a rating matrix for an easier explanation:

   ```python
   # Create user-item matrix
   user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
   print(user_item_matrix.head())
   ```

---

### **Step 6: Build Collaborative Filtering Model**
1. Use the surprise library's `SVD` (Singular Value Decomposition) algorithm to build the collaborative filter.

   ```python
   # Prepare the data for the surprise library
   reader = Reader(rating_scale=(0.5, 5.0))
   data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

   # Build the training set
   trainset = data.build_full_trainset()

   # Train the SVD model
   model = SVD()
   model.fit(trainset)
   ```

---

### **Step 7: Create a Movie Similarity Function**
1. To recommend movies similar to a given title, we need to calculate cosine similarity between movies based on collaborative filtering results.

   ```python
   # Create a function to estimate all user ratings for a specific movie
   def get_movie_predictions(model, title, movies, ratings, n_recommendations=10):
       # Find the movieId for the given title
       movie = movies[movies['title'].str.contains(title, case=False)].iloc[0]
       movie_id = movie['movieId']

       # Get the predictions for all users for this movie
       predictions = defaultdict(float)
       for user_id in ratings['userId'].unique():
           predictions[user_id] = model.predict(user_id, movie_id).est

       # Get movies sorted by predicted rating
       sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
       result = movies[movies['movieId'].isin([x[0] for x in sorted_movies])]
       
       return result[['title', 'movieId']]

   # Example usage
   recommendations = get_movie_predictions(model, "Toy Story", movies, ratings, n_recommendations=10)
   print(recommendations)
   ```
---
