# Encryption-of-Biometric-Traits
![image](https://github.com/user-attachments/assets/d3c9441d-a0f0-495e-8f0f-6b1b388d2d4e)
Overview
This project demonstrates a secure encryption mechanism based on biometric features and AES (Advanced Encryption Standard). The system utilizes biometric feature extraction, quantization techniques, and AES-GCM (Galois/Counter Mode) encryption to securely encrypt and decrypt user data. The biometric key is dynamically derived from the user’s biometric data (e.g., iris or facial features), ensuring robust and personalized security.
The project includes:
•	Data loading, preprocessing, and visualization for MMU Iris Database.
•	Feature extraction using pre-trained CNN models (VGG16).
•	Principal Component Analysis (PCA) for face attributes.
•	Quantization-based key generation for AES encryption.
•	Streamlit frontend for user interaction and real-time encryption/decryption.
________________________________________
Features
1.	Iris Image Preprocessing:
o	Loads grayscale iris images, resizes, and normalizes them.
o	Visualizes left and right iris samples.
2.	Feature Extraction:
o	Utilizes VGG16 (pre-trained CNN) for extracting features from preprocessed iris images.
o	Reduces dimensions of facial attributes using PCA.
3.	Biometric Key Generation:
o	Quantizes combined biometric features into binary keys.
o	Converts the binary key to AES-compatible byte format.
4.	AES-GCM Encryption and Decryption:
o	Encrypts user-provided text data using a biometric-derived AES key.
o	Ensures data integrity using authentication tags.
5.	Streamlit Web Interface:
o	User-friendly frontend for encryption and decryption.
o	Options to upload biometric features or dynamically generate synthetic ones.
________________________________________
Modules and Libraries

The project utilizes the following Python libraries:
Data Handling and Visualization
•	pandas: Data loading and manipulation.
•	numpy: Numerical computations.
•	matplotlib: Data visualization.
•	seaborn: Enhanced visualizations.
•	OpenCV (cv2): Image processing.
Machine Learning and Dimensionality Reduction
•	tensorflow.keras: Pre-trained VGG16 model for feature extraction.
•	sklearn: StandardScaler and PCA for face attributes.
Encryption and Security
•	pycryptodome: AES-GCM encryption and decryption.
•	Cryptodome.Util.Padding: Padding for AES encryption.
Web Interface
•	streamlit: Interactive frontend for real-time encryption/decryption.
________________________________________
Algorithm Workflow

1. Iris Image Processing
•	Load and preprocess images from the MMU Iris Database.
•	Resize images to 128x128 and normalize pixel values.
2. Feature Extraction
•	Preprocess images for the VGG16 model (224x224 RGB format).
•	Extract features from the images using the VGG16 pre-trained model.
•	Flatten the features for further use.
3. Face Attribute Processing
•	Load CelebA facial attributes dataset (CSV).
•	Standardize attributes and apply PCA to reduce dimensions.
4. Biometric Key Generation
•	Combine extracted iris and face features.
•	Normalize and quantize features into binary keys (16 bins by default).
•	Convert the binary string to an AES-compatible 256-bit key.
5. AES-GCM Encryption and Decryption
•	Use the biometric key to encrypt user-provided text.
•	Generate a nonce and authentication tag for integrity.
•	Decrypt and verify the data using the same key.
6. Streamlit Frontend
•	Allow users to upload biometric features or generate them dynamically.
•	Enable data encryption and decryption via AES-GCM.
________________________________________
Use Case
Secure Data Encryption with Biometric Keys
This project can be used in:
1.	Identity-Based Security Systems: Secure encryption keys generated from unique biometric features.
2.	Healthcare and Biometrics: Encrypting sensitive patient data using iris or facial features.
3.	Authentication Systems: Biometric-derived cryptographic keys for secure authentication.
4.	Confidential Communication: Protecting confidential messages with personalized encryption keys.
________________________________________

Setup Instructions
Follow these steps to run the project:
1.	Clone the Repository:
git clone https://github.com/YourUsername/Encryption-of-Biometric-Traits.git
cd Encryption-of-Biometric-Traits
2.	Install Required Libraries: Run the following command to install dependencies:
!pip install -r requirements.txt
3.	Run the Streamlit App:
!streamlit run app.py
4.	Access the App: Go to http://localhost:8501 or the public URL generated if using Colab.
________________________________________
Summary
This project provides an innovative encryption system by leveraging biometric features for AES key generation. The combination of deep learning-based feature extraction, PCA for dimensionality reduction, and quantization techniques enables robust and secure encryption.
The integration with Streamlit makes the project interactive and easy to use, offering a seamless way to encrypt and decrypt user data securely.
