import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from Cryptodome.Cipher import AES

# Paths
IRIS_IMAGES_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/MMU-Iris-Database"
FACE_CSV_PATH = r"C:/Users/KAREN/Documents/CAPSTONE/Datasets/list_attr_celeba.csv"

# --- Functions from visualization.py ---
# Load Precomputed Features
def load_precomputed_features():
    with open("combined_features.pkl", "rb") as f:
        return pickle.load(f)

def load_explained_variance():
    with open("explained_variance.pkl", "rb") as f:
        return pickle.load(f)

# Load and preprocess Face Attributes
def load_face_attributes(csv_path):
    face_df = pd.read_csv(csv_path)
    face_df.replace({-1: 0}, inplace=True)  # Convert -1 to 0
    for col in face_df.columns:
        if face_df[col].dtype == "object":
            try:
                face_df[col] = pd.to_numeric(face_df[col], errors="coerce")
            except Exception:
                pass
    return face_df

# Load Iris Images
def load_random_iris_images(data_path, num_samples=5, image_size=(128, 128)):
    images = []
    for subject_folder in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject_folder)
        if os.path.isdir(subject_path):
            for lr_folder in os.listdir(subject_path):
                lr_path = os.path.join(subject_path, lr_folder)
                if os.path.isdir(lr_path):
                    for file in os.listdir(lr_path):
                        if file.lower().endswith('.bmp'):
                            img_path = os.path.join(lr_path, file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, image_size)
                                images.append(img_resized)
    if len(images) > num_samples:
        np.random.shuffle(images)
        images = images[:num_samples]
    return images

# Quantization Key Extraction
def quantization_key_extraction(biometric_features, num_bins=16):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(biometric_features)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    quantized_indices = np.digitize(normalized_features, bins=bin_edges) - 1
    bin_width = int(np.ceil(np.log2(num_bins)))
    binary_keys = []
    for indices in quantized_indices:
        binary_key = ''.join([format(index, f'0{bin_width}b') for index in indices])
        binary_keys.append(binary_key)
    return binary_keys

# Convert Binary Key to Bytes
def binary_to_bytes(binary_key, key_length=256):
    truncated_key = binary_key[:key_length]
    return int(truncated_key, 2).to_bytes(key_length // 8, byteorder='big')

# AES Encrypt/Decrypt
def aes_encrypt_decrypt(text, aes_key, mode="encrypt", nonce=None, tag=None):
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce) if nonce else AES.new(aes_key, AES.MODE_GCM)
    if mode == "encrypt":
        encrypted_data, tag = cipher.encrypt_and_digest(text.encode())
        return encrypted_data, cipher.nonce, tag
    elif mode == "decrypt":
        return cipher.decrypt_and_verify(text, tag).decode()

# --- Main Streamlit App ---
st.title("🔒 Biometric Key-Based AES Encryption")
st.markdown("### A Streamlit app to demonstrate AES encryption using biometric keys.")

# Sidebar for Options
st.sidebar.title("Options")
option = st.sidebar.radio("Choose an Option:", ["Generate Key Dynamically", "Encrypt a File", "Visualization of Data"])

if option == "Generate Key Dynamically":
    # Sidebar input for number of bins
    st.sidebar.subheader("Quantization Settings")
    num_bins = st.sidebar.slider("Number of Bins for Quantization:", 8, 32, 16)

    # Generate Key Dynamically
    if st.button("Generate Key Dynamically"):
        combined_features = load_precomputed_features()
        binary_keys = quantization_key_extraction(combined_features, num_bins=num_bins)
        backend_key = binary_keys[0]
        st.session_state['binary_key'] = backend_key
        st.success("Biometric Key Generated Dynamically!")
        st.code(f"Key (Binary): {backend_key[:256]}")

    # Encryption Section
    st.subheader("🔐 Encrypt Your Data")
    user_input = st.text_area("Enter Data to Encrypt", "")
    if st.button("Encrypt Data"):
        if 'binary_key' in st.session_state:
            aes_key = binary_to_bytes(st.session_state['binary_key'])
            encrypted_data, nonce, tag = aes_encrypt_decrypt(user_input, aes_key, mode="encrypt")
            st.session_state['encrypted_data'] = encrypted_data
            st.session_state['nonce'] = nonce
            st.session_state['tag'] = tag
            st.success("Data Encrypted Successfully!")
            st.code(f"Encrypted Data (Hex): {encrypted_data.hex()}")

    # Decryption Section
    st.subheader("🔓 Decrypt Your Data")
    if st.button("Decrypt Data"):
        if 'encrypted_data' in st.session_state:
            aes_key = binary_to_bytes(st.session_state['binary_key'])
            decrypted_data = aes_encrypt_decrypt(st.session_state['encrypted_data'], aes_key, mode="decrypt",
                                                 nonce=st.session_state['nonce'], tag=st.session_state['tag'])
            st.success("Data Decrypted Successfully!")
            st.code(f"Decrypted Data: {decrypted_data}")

elif option == "Encrypt a File":
    st.subheader("📄 File Encryption")

    # File Upload
    uploaded_file = st.file_uploader("Upload a File to Encrypt", type=["txt", "csv", "json"])
    if uploaded_file:
        file_content = uploaded_file.read()
        st.text("File Uploaded Successfully!")
        st.text_area("File Content", file_content.decode(), height=200)

        # Encrypt Uploaded File
        if st.button("Encrypt File"):
            if 'binary_key' in st.session_state:
                aes_key = binary_to_bytes(st.session_state['binary_key'])
                encrypted_data, nonce, tag = aes_encrypt_decrypt(file_content.decode(), aes_key, mode="encrypt")
                st.session_state['encrypted_file'] = encrypted_data
                st.session_state['nonce'] = nonce
                st.session_state['tag'] = tag
                st.success("File Encrypted Successfully!")

                # Enable Download of Encrypted File
                st.download_button(
                    label="Download Encrypted File",
                    data=encrypted_data,
                    file_name="encrypted_file.bin",
                    mime="application/octet-stream"
                )

elif option == "Visualization of Data":
    st.subheader("📊 Data Visualization for Biometric Features")
    st.markdown("Explore raw and processed biometric features from iris and face datasets.")

    visualization_option = st.sidebar.radio(
        "Choose a Feature Type to Visualize:",
        ["Raw Iris Images", "Face Attributes", "PCA Analysis", "Explained Variance", "CNN Features"]
    )

    if visualization_option == "Raw Iris Images":
        iris_images = load_random_iris_images(IRIS_IMAGES_PATH)
        st.markdown("### Randomly Selected Iris Images")
        cols = st.columns(len(iris_images))
        for i, img in enumerate(iris_images):
            cols[i].image(img, caption=f"Iris {i+1}", use_container_width=True)

    elif visualization_option == "Face Attributes":
        face_data = load_face_attributes(FACE_CSV_PATH)
        st.markdown("### Number of Positive Values for Each Face Attribute")
        positive_counts = face_data.sum(axis=0).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=positive_counts.values, y=positive_counts.index, palette="viridis")
        plt.title("Number of Positive Values for Each Face Attribute")
        plt.xlabel("Count")
        plt.ylabel("Attributes")
        st.pyplot(plt.gcf())

    elif visualization_option == "PCA Analysis":
        explained_variance_ratio = load_explained_variance()
        st.markdown("### Scree Plot of PCA Explained Variance")
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid()
        st.pyplot(plt.gcf())

    elif visualization_option == "Explained Variance":
        explained_variance_ratio = load_explained_variance()
        cumulative_variance = np.cumsum(explained_variance_ratio)
        st.markdown("### Cumulative Explained Variance")
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='orange')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        st.pyplot(plt.gcf())

    elif visualization_option == "CNN Features":
        combined_features = load_precomputed_features()
        st.markdown("### Heatmap of Extracted CNN Features")
        num_samples = st.slider("Number of Samples to Visualize:", min_value=1, max_value=combined_features.shape[0], value=10)
        num_features = st.slider("Number of Features to Visualize:", min_value=1, max_value=combined_features.shape[1], value=50)
        random_samples = np.random.choice(combined_features.shape[0], num_samples, replace=False)
        random_features = np.random.choice(combined_features.shape[1], num_features, replace=False)
        subset_features = combined_features[np.ix_(random_samples, random_features)]
        plt.figure(figsize=(10, 8))
        sns.heatmap(subset_features, cmap="viridis", annot=False)
        plt.title("Heatmap of CNN-Extracted Features (Subset)")
        plt.xlabel("Selected Features")
        plt.ylabel("Selected Samples")
        st.pyplot(plt.gcf())

# Footer Section
st.markdown("---")
st.markdown("© 2024 Karen Rena C. All rights reserved.")
