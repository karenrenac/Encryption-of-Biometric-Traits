import pickle
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Paths for datasets
IRIS_IMAGES_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/MMU-Iris-Database"
FACE_CSV_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/list_attr_celeba.csv"

# --- Backend Functions ---
def load_iris_data(data_path, image_size=(128, 128)):
    images = []
    for subject_folder in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject_folder)
        if os.path.isdir(subject_path):
            for lr_folder in os.listdir(subject_path):  # 'left' and 'right' folders
                lr_path = os.path.join(subject_path, lr_folder)
                if os.path.isdir(lr_path):
                    for file in os.listdir(lr_path):
                        if file.lower().endswith('.bmp'):  # Only load BMP images
                            img_path = os.path.join(lr_path, file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, image_size)
                                images.append(img_resized)
    images = np.array(images) / 255.0  # Normalize pixel values to range [0, 1]
    return images

def preprocess_for_cnn(images, target_size=(224, 224)):
    preprocessed_images = []
    for img in images:
        img = (img * 255).astype(np.uint8)  # Convert float64 to uint8
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        preprocessed_images.append(img_rgb)
    preprocessed_images = np.array(preprocessed_images)
    print(f"Number of preprocessed iris images: {preprocessed_images.shape[0]}, Image shape: {preprocessed_images.shape[1:]}")
    return preprocessed_images

def extract_features(model, images):
    features = model.predict(images, verbose=1)
    features = features.reshape(features.shape[0], -1)
    print(f"Number of extracted iris features: {features.shape}")
    return features

def preprocess_face_csv(csv_path):
    face_df = pd.read_csv(csv_path)
    numeric_face_df = face_df.drop(columns=['image_id']).replace({-1: 0})
    return numeric_face_df

def apply_pca_to_face_data(face_data, variance_threshold=0.95):
    # Standardize the data
    scaler = StandardScaler()
    standardized_face_data = scaler.fit_transform(face_data)

    # Fit PCA
    pca = PCA()
    pca_features = pca.fit_transform(standardized_face_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Determine the number of components needed to meet the variance threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Re-run PCA with the determined number of components
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(standardized_face_data)

    # Print the total variance retained
    total_variance_retained = cumulative_variance[n_components - 1]
    print(f"Total variance retained: {total_variance_retained * 100:.2f}%")
    
    return pca_features, explained_variance_ratio[:n_components], n_components

def combine_features(iris_features, face_features, num_samples=450):
    np.random.seed(42)  # For reproducibility
    sampled_face_features = face_features[np.random.choice(face_features.shape[0], num_samples, replace=False)]
    combined = np.hstack((iris_features, sampled_face_features))
    print(f"Number of combined features: {combined.shape}")
    return combined

# --- Preprocessing Workflow ---
print("Loading and processing Iris images...")
iris_images = load_iris_data(IRIS_IMAGES_PATH)
processed_images = preprocess_for_cnn(iris_images)

print("Extracting Iris features with VGG16...")
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=vgg16_base.input, outputs=vgg16_base.output)
iris_features = extract_features(feature_extractor, processed_images)

print("Processing Face attributes CSV...")
face_data = preprocess_face_csv(FACE_CSV_PATH)

# Apply dynamic PCA to meet 95% explained variance
face_features, explained_variance_ratio, n_components = apply_pca_to_face_data(face_data, variance_threshold=0.95)

print(f"PCA completed. Number of components to retain 95% variance: {n_components}")

# Save PCA explained variance ratio as .pkl file
with open("explained_variance.pkl", "wb") as f:
    pickle.dump(explained_variance_ratio, f)

print("Combining Iris and Face features...")
combined_features = combine_features(iris_features, face_features)

# Save combined features as a .pkl file
with open("combined_features.pkl", "wb") as f:
    pickle.dump(combined_features, f)

print("Features and explained variance saved.")
