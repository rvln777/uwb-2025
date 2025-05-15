### **Collaborative Filtering Movie Recommendation System**

---

### **Step 1: Install Required Libraries**
To start, ensure you have Python installed with necessary libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

### **Step 2: Download the MovieLens Dataset**
The MovieLens dataset provides movie ratings by users.

1. Download the dataset from [MovieLens website](https://grouplens.org/datasets/movielens/).
2. Choose the size of the dataset (e.g., `ml-latest-small.zip`).
3. Extract the zip file to your working directory.

---

### **Step 3: Load and Explore the Dataset**
Let’s load the data and perform an initial check:

```python
import pandas as pd

# Load ratings.csv
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Load movies.csv
movies = pd.read_csv('ml-latest-small/movies.csv')

# Display sample data
print(ratings.head())
print(movies.head())
```

Verify the structure of ratings and movies data:
- **Ratings**: userId, movieId, rating, timestamp
- **Movies**: movieId, title, genres

---

### **Step 4: Preprocess the Data**
Clean and reshape the data for collaborative filtering.

```python
# Drop unnecessary columns
ratings = ratings.drop('timestamp', axis=1)

# Merge ratings and movies on movieId for better exploration
movie_data = pd.merge(ratings, movies, on='movieId')

print(movie_data.head())
```

---

### **Step 5: Collaborative Filtering Overview**
Collaborative filtering relies on interactions (ratings) between users and items (movies). Two main techniques:
1. **User-based Collaborative Filtering**: Recommends movies based on similar users.
2. **Item-based Collaborative Filtering**: Recommends movies based on similar items.

In this example, we’ll use Matrix Factorization (e.g., Singular Value Decomposition, or SVD) to implement collaborative filtering.

---

### **Step 6: Create a User-Item Interaction Matrix**
Prepare a matrix where rows represent users and columns represent movies.

```python
# Create a pivot table with users as rows and movies as columns
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill NA values with 0 since ratings are sparse
user_movie_matrix = user_movie_matrix.fillna(0)

print(user_movie_matrix.head())
```

---

### **Step 7: Implement Collaborative Filtering with SVD**
Singular Value Decomposition (SVD) is used to decompose the matrix into latent factors.

```python
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Initialize SVD
svd = TruncatedSVD(n_components=50)  # Reduce dimensions
latent_matrix = svd.fit_transform(user_movie_matrix)

# Compute explained variance ratio
print("Explained variance ratio:", svd.explained_variance_ratio_.sum())
```

---

### **Step 8: Predict Ratings**
Reconstruct the interaction matrix from latent factors to predict user ratings.

```python
# Get reconstructed predictions by multiplying latent matrices
predicted_ratings = np.dot(latent_matrix, svd.components_)

# Convert predictions back to a Pandas DataFrame for better compatibility
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
print(predicted_ratings_df.head())
```

---

### **Step 9: Recommend Movies**
Recommend top movies for a specific user.

```python
def recommend_movies(user_id, original_matrix, predicted_matrix, movies, num_recommendations=5):
    # Find user row
    user_ratings = original_matrix.loc[user_id]
    
    # Find user's predicted ratings
    user_predictions = predicted_matrix.loc[user_id]
    
    # Filter out movies that the user has already rated
    unrated_movies = user_ratings[user_ratings == 0].index
    predicted_unrated = user_predictions[unrated_movies]
    
    # Recommend movies with the highest predicted ratings
    top_movies = predicted_unrated.sort_values(ascending=False).head(num_recommendations)
    
    # Convert movie ids to movie titles
    recommended_titles = movies[movies['movieId'].isin(top_movies.index)]['title']
    return recommended_titles

# Recommend movies for user 1
recommendations = recommend_movies(1, user_movie_matrix, predicted_ratings_df, movies)
print("Recommended Movies for User 1:")
print(recommendations)
```

---

### **Step 10: Evaluate the Recommendation System**
Evaluate the performance using techniques like Root Mean Square Error (RMSE).

```python
from sklearn.metrics import mean_squared_error
from math import sqrt

# Flatten matrices for comparison
actual = user_movie_matrix.values.flatten()
predicted = predicted_ratings.flatten()

# Filter out unrated (zero) values
mask = actual != 0
actual = actual[mask]
predicted = predicted[mask]

# Calculate RMSE
rmse = sqrt(mean_squared_error(actual, predicted))
print(f"RMSE: {rmse}")
```
