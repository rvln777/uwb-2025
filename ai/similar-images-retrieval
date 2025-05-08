# Task: find most simialr images from image query

### **Step 1: Set Up the Environment**
1. Install the required libraries (TensorFlow, NumPy, matplotlib, scikit-learn, etc.).
    ```bash
    pip install tensorflow matplotlib numpy scikit-learn
    ```

2. Import the necessary libraries in your Python script or Jupyter Notebook:
    ```python
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    ```

---

### **Step 2: Load Pre-Trained CNN Model**
1. Load a pre-trained CNN, such as VGG16, with weights trained on ImageNet. Use it without the final classification layers to get feature representations.
    ```python
    # Load VGG16 model excluding the top (fully connected) layers
    model = VGG16(weights='imagenet', include_top=False)  # Change the model based on your needs
    model.summary()
    ```

2. Note: The `include_top=False` argument excludes the fully connected layers at the top, allowing you to get the "deep features" (feature maps) from the convolutional layers.

---

### **Step 3: Prepare the Dataset**
1. Collect a dataset of images for retrieval. Save the images in a folder structure (e.g., `images/` folder).

2. Preprocess the images and resize them to the model's required input dimensions (224x224 for VGG16).
    ```python
    def preprocess_image(image_path, target_size=(224, 224)):
        # Load and preprocess a single image
        image = load_img(image_path, target_size=target_size)  # Resize image
        image = img_to_array(image)                           # Convert to array
        image = np.expand_dims(image, axis=0)                 # Expand dims for batch size
        image = preprocess_input(image)                       # Preprocess based on model
        return image
    ```

3. Create a list of all the image file paths in the dataset folder:
    ```python
    data_dir = 'images/'  # Replace with your dataset folder path
    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".jpg")]
    ```

---

### **Step 4: Extract Deep Features**
1. Write a function to pass images through the CNN and extract deep features.
    ```python
    def extract_features(image_paths, model):
        features = []
        for image_path in image_paths:
            image = preprocess_image(image_path)  # Preprocess image
            feature = model.predict(image)       # Extract features using pretrained model
            features.append(feature.flatten())  # Flatten to 1D feature vector
        return np.array(features)
    ```

2. Extract features for all dataset images:
    ```python
    features = extract_features(image_paths, model)
    print(f"Extracted features shape: {features.shape}")
    ```

3. Optionally, normalize the features to have unit norm, which helps in similarity search:
    ```python
    features = normalize(features)
    ```

---

### **Step 5: Build an Image Retrieval System**
1. Define a function to compute similarity between feature vectors. We'll use **cosine similarity**:
    ```python
    def find_similar_images(query_image_path, image_paths, features, model, top_k=5):
        # Extract features for the query image
        query_image = preprocess_image(query_image_path)
        query_feature = model.predict(query_image).flatten()
        query_feature = normalize(query_feature[None, :])  # Normalize and reshape
        
        # Compute cosine similarity between query feature and dataset features
        similarities = cosine_similarity(query_feature, features)[0]
        
        # Sort by similarity score
        indices = np.argsort(similarities)[::-1][:top_k]  # Indices of top-k similar images
        similar_images = [(image_paths[i], similarities[i]) for i in indices]
        
        return similar_images
    ```

2. Test the function with a query image:
    ```python
    query_image_path = 'images/query.jpg'  # Replace with your query image path
    similar_images = find_similar_images(query_image_path, image_paths, features, model, top_k=5)
    
    # Print results
    for img_path, score in similar_images:
        print(f"Image: {img_path}, Similarity: {score:.4f}")
    ```

---

### **Step 6: Display the Results**
1. Plot the query image and its top-k results:
    ```python
    def display_similar_images(query_image_path, similar_images):
        # Display query image
        query_img = load_img(query_image_path, target_size=(224, 224))
        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(similar_images) + 1, 1)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis("off")
        
        # Display similar images
        for i, (img_path, score) in enumerate(similar_images):
            img = load_img(img_path, target_size=(224, 224))
            plt.subplot(1, len(similar_images) + 1, i + 2)
            plt.imshow(img)
            plt.title(f"Score: {score:.4f}")
            plt.axis("off")
        plt.show()
    
    # Display top-5 most similar images
    display_similar_images(query_image_path, similar_images)
    ```

---

### **Step 7: Extend the System (Optional)**
1. Replace VGG16 with other pre-trained models like ResNet, EfficientNet, or MobileNet for better performance.
2. Use a dimensionality reduction technique (e.g., PCA or t-SNE) for reducing feature dimensionality.
3. Consider implementing approximate nearest neighbor (ANN) algorithms (e.g., FAISS) for faster similarity search in large datasets.
