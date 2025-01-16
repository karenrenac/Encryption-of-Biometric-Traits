# Encryption-of-Biometric-Traits

![image](https://github.com/user-attachments/assets/242e8264-9c3b-431d-a237-bc20bec7f6f1)

## Overview
This project demonstrates a secure encryption mechanism based on biometric features and AES (Advanced Encryption Standard). The system utilizes biometric feature extraction, quantization techniques, and AES-GCM (Galois/Counter Mode) encryption to securely encrypt and decrypt user data. The biometric key is dynamically derived from the user’s biometric data (e.g., iris or facial features), ensuring robust and personalized security.
________________________________________
## Setup Instructions
Follow these steps to run the project:
1. Clone the Repository:
```
git clone https://github.com/karenrenac/Encryption-of-Biometric-Traits.git
cd Encryption-of-Biometric-Traits-using-AES-Encryption
```
2. Precompute the biometric features by running:
```
python precompute_features.py
```
This script processes the iris and face datasets, extracts features, and saves them in a combined_features.pkl file. The explained_variance.pkl file will be used for data visualization

3. Run the Streamlit App:
```
streamlit run main.py
```
4. Access the App at http://localhost:8501 in your browser.


## Features
1.	Feature Extraction:
*	Utilizes VGG16 (pre-trained CNN) for extracting features from preprocessed iris images.
*	Reduces dimensions of facial attributes using PCA.

2.	Biometric Key Generation:
*	Quantizes combined biometric features into binary keys.
*	Converts the binary key to AES-compatible byte format.

3.	AES-GCM Encryption and Decryption:
*	Encrypts user-provided text data or file using a biometric-derived AES key.
*	Ensures data integrity using authentication tags.

4.	Streamlit Web Interface:
*	User-friendly frontend for encryption and decryption.
*	Options to upload biometric features or dynamically generate synthetic ones.
________________________________________
## Modules and Libraries

The project utilizes the following Python libraries:

* Streamlit: For the interactive frontend.
* NumPy: For numerical operations.
* scikit-learn: For preprocessing and PCA.
* TensorFlow: For feature extraction using the VGG16 model.
* PyCryptodomex: For AES encryption and decryption.
* OpenCV: For image processing.
* Pandas: For handling and preprocessing CSV data.

________________________________________
## Algorithm Workflow

Preprocessing (precompute_features.py)
1. Iris Data:
* Load and preprocess .bmp images.
* Resize and normalize images.
* Convert grayscale to RGB.
2. Feature Extraction:
* Use VGG16 CNN model to extract high-dimensional features.
3. Face Data:
* Load and preprocess the CelebA CSV file.
* Replace missing values, standardize, and apply PCA for dimensionality reduction.
4. Combine Features:
* Merge iris and face features into a single feature vector.
* Save the combined features to a combined_features.pkl file.

Encryption Workflow (main.py)
1. Frontend:
* User selects options (key generation, encryption, decryption).
* User inputs data or uploads files.
2. Key Generation:
* Quantize the biometric features to generate binary keys.
* Convert binary keys to AES-compatible byte keys.
3. Encryption:
* Encrypt user data using AES-GCM.
* Return encrypted data, nonce, and tag.
4. Decryption:
* Decrypt the encrypted data using AES-GCM and validate integrity.
________________________________________
## Use Case

Secure Data Encryption with Biometric Keys

This project can be used in:
1.	Identity-Based Security Systems: Secure encryption keys generated from unique biometric features.
2.	Healthcare and Biometrics: Encrypting sensitive patient data using iris or facial features.
3.	Authentication Systems: Biometric-derived cryptographic keys for secure authentication.
4.	Confidential Communication: Protecting confidential messages with personalized encryption keys.

________________________________________
## Summary

This project is a cutting-edge demonstration of combining biometrics and cryptography to create a secure encryption system. By dynamically generating keys from biometric data, it eliminates the need for manually managed cryptographic keys, enhancing security and usability. The user-friendly Streamlit interface makes it accessible to both developers and non-technical users.
