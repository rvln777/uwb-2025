# Movie recommendation system

# Task
Create an applicaation which takes movie title as input and returns list of movies similar to it.

Here’s a step-by-step guide to implement a movie recommendation system using collaborative filtering on the MovieLens dataset. Collaborative filtering relies on user-item interaction (e.g., user ratings) to make recommendations by identifying patterns in user behavior.

---

### **Step 1: Set Up Your Environment**
1. **Install Dependencies**:
   You’ll need Python and libraries such as pandas, NumPy, scikit-learn, and optionally TensorFlow or PyTorch. Install using pip:
   ```bash
   pip install pandas numpy scikit-learn
   ```

2. **Download the MovieLens Dataset**:
   Visit [MovieLens Dataset](https://grouplens.org/datasets/movielens/) and download a suitable size dataset (e.g., ml-latest-small or ml-1m). Extract the dataset.

---

### **Step 2: Understand the Dataset**
1. The dataset typically includes:
   - **`movies.csv`**: Movie IDs, titles, and genres.
   - **`ratings.csv`**: User IDs, movie IDs, ratings, and timestamps.
   - **`links.csv`** (optional): Links to external movie information (e.g., IMDB).
   
2. Load the data using pandas:
   ```python
   import pandas as pd
   
   movies = pd.read_csv('movies.csv')
   ratings = pd.read_csv('ratings.csv')
   print(movies.head())
   print(ratings.head())
   ```

---

### **Step 3: Preprocess the Data**
1. **Drop irrelevant columns**:
   Remove unnecessary columns (e.g., genres, timestamps) from the dataset.

2. **Handle missing values**:
   Check if there are any missing values in the dataset and handle them:
   ```python
   print(ratings.isnull().sum())
   ratings.dropna(inplace=True)
   ```

3. Create a user-item matrix:
   Pivot the ratings data to create a matrix where rows are users and columns are movie IDs:
   ```python
   user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
   ```

---

### **Step 4: Implement Collaborative Filtering**
Collaborative filtering approaches can be:
   - **User-Based**: Recommends movies based on similar users.
   - **Item-Based**: Recommends movies similar to the one the user interacted with.

For this example, we’ll use item-based collaborative filtering.

#### **Approach: Cosine Similarity**
1. Compute the similarity between movies based on their ratings:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   import numpy as np
   
   # Fill missing values with 0
   user_movie_matrix_filled = user_movie_matrix.fillna(0)
   movie_similarity = cosine_similarity(user_movie_matrix_filled.T)  # Transpose to compare movies
   
   # Create a DataFrame for easier handling
   movie_similarity_df = pd.DataFrame(movie_similarity, 
                                       index=user_movie_matrix.columns, 
                                       columns=user_movie_matrix.columns)
   ```

2. Verify the similarity matrix:
   ```python
   print(movie_similarity_df.head())
   ```

---

### **Step 5: Create a Recommendation Function**
Given a movie title, find similar movies based on the computed similarity matrix.

1. Map movie IDs to titles:
   ```python
   movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
   movie_title_to_id = dict(zip(movies['title'], movies['movieId']))
   ```

2. Create the function:
   ```python
   def recommend_movies(movie_title, similarity_df, movie_title_to_id, movie_id_to_title, n=10):
       if movie_title not in movie_title_to_id:
           print("Movie not found!")
           return []
       
       movie_id = movie_title_to_id[movie_title]
       similar_movie_ids = similarity_df[movie_id].sort_values(ascending=False).head(n+1).index.tolist()
       
       # Exclude the target movie itself from recommendations
       recommendations = [movie_id_to_title[m_id] for m_id in similar_movie_ids if m_id != movie_id]
       return recommendations
   ```

3. Test the function:
   ```python
   movie_title = "Toy Story (1995)"  # Replace with a movie title from your dataset
   recommendations = recommend_movies(movie_title, movie_similarity_df, movie_title_to_id, movie_id_to_title, n=5)
   print(f"Movies similar to '{movie_title}':")
   print(recommendations)
   ```

---
