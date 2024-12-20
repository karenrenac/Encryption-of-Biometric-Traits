import pickle
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import os

# Paths for datasets
IRIS_IMAGES_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/MMU-Iris-Database"
FACE_CSV_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/list_attr_celeba.csv"

# --- Backend Functions ---
def load_iris_data(data_path, image_size=(128, 128)):
    """
    Load and preprocess iris images from nested dataset structure.
    """
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
    """
    Preprocess grayscale images to match pre-trained CNN input requirements.
    """
    preprocessed_images = []
    for img in images:
        img = (img * 255).astype(np.uint8)  # Convert float64 to uint8
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        preprocessed_images.append(img_rgb)
    return np.array(preprocessed_images)

def extract_features(model, images):
    """
    Extract features using a pre-trained CNN model.
    """
    features = model.predict(images, verbose=1)
    return features.reshape(features.shape[0], -1)

def preprocess_face_csv(csv_path):
    """
    Preprocess face CSV file: replace -1 with 0, drop 'image_id', and standardize data.
    """
    face_df = pd.read_csv(csv_path)
    numeric_face_df = face_df.drop(columns=['image_id']).replace({-1: 0})
    return numeric_face_df

def apply_pca_to_face_data(face_data, n_components=10):
    """
    Apply PCA to the face attributes data.
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(face_data)
    return pca_features

def combine_features(iris_features, face_features, num_samples=450):
    """
    Combine iris and face features, ensuring the sample size matches.
    """
    np.random.seed(42)  # For reproducibility
    sampled_face_features = face_features[np.random.choice(face_features.shape[0], num_samples, replace=False)]
    return np.hstack((iris_features, sampled_face_features))

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
face_features = apply_pca_to_face_data(face_data, n_components=10)

print("Combining Iris and Face features...")
combined_features = combine_features(iris_features, face_features)

# Save combined features as a .pkl file
with open("combined_features.pkl", "wb") as f:
    pickle.dump(combined_features, f)

print("Features saved to 'combined_features.pkl'")
