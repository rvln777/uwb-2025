### **Step 1: Preparation**
Install the necessary Python libraries if you haven't already:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

### **Step 2: Import Libraries**
Start by importing the required libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
```

---

### **Step 3: Load and Explore the Dataset**
You can either use an existing dataset of restaurant reviews or create one manually. For demonstration, let's assume you are working with a CSV file named `restaurant_reviews.csv` containing columns `'Review'` (text) and `'Sentiment'` (binary: 1 for positive, 0 for negative).

```python
# Load the dataset
df = pd.read_csv('restaurant_reviews.csv')

# Display the top rows of the dataset
print(df.head())

# Check for null values
print(df.isnull().sum())

# Explore class distribution
print(df['Sentiment'].value_counts())
```

---

### **Step 4: Preprocess the Data**
Preprocess the text data using `CountVectorizer` (convert text to numerical features).
```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the review text
X = vectorizer.fit_transform(df['Review']).toarray()

# Target variable (Sentiment)
y = df['Sentiment'].values
```

---

### **Step 5: Split the Data**
Divide the data into a training set and testing set. Typically, 80% of data for training and 20% for testing is used.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
```

---

### **Step 6: Build the Logistic Regression Model**
Train the classifier using Scikit-learn's `LogisticRegression`.
```python
# Initialize the logistic regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)
```

---

### **Step 7: Make Predictions**
Predict sentiment for the test data.
```python
# Predict on test data
y_pred = log_reg.predict(X_test)

# Print example predictions
print("Predicted Sentiments:", y_pred[:5])
```

---

### **Step 8: Evaluate the Model**
#### **a) Accuracy**
Evaluate the accuracy of the model.
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### **b) Confusion Matrix**
Analyze the confusion matrix to see how the model performs on each class.
```python
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

#### **c) ROC Curve and AUC**
Compute and plot the Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).
```python
# Get probabilities for the positive class
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Compute AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.2f}")

# Plot the ROC curve
plt.plot(fpr, tpr, color='orange', label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

### **Step 9: Test the Model on New Reviews**
Optional: Test the model with new, unseen reviews.
```python
new_reviews = ["The food was amazing!", "Terrible service, I won't come back."]
new_reviews_transformed = vectorizer.transform(new_reviews)

# Predict sentiment
new_predictions = log_reg.predict(new_reviews_transformed)
print("Predicted Sentiments for new reviews:", new_predictions)
```

----

### **Step 10: Create application**
Input: comment
Output: positive, negative

---

### **Step 11: Wrap Up**
You have successfully built and evaluated a logistic regression classifier to predict restaurant review sentiment. Use interpretability tools or advanced text preprocessing techniques—such as TF-IDF or word embeddings—for better performance if the dataset is large or complex.

