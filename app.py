import streamlit as st
import numpy as np
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from sklearn.preprocessing import MinMaxScaler
import os

# --- Helper Functions ---
def display_title():
    st.title("🔒 Biometric Key-Based AES Encryption")
    st.subheader("Encrypt Your Data Using AES-GCM and Biometric Key")

def aes_encrypt_decrypt(text, aes_key, mode="encrypt", nonce=None, tag=None):
    """
    AES encryption or decryption in GCM mode.
    """
    try:
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce) if nonce else AES.new(aes_key, AES.MODE_GCM)
        if mode == "encrypt":
            encrypted_data, tag = cipher.encrypt_and_digest(text.encode())
            return encrypted_data, cipher.nonce, tag
        elif mode == "decrypt":
            return cipher.decrypt_and_verify(text, tag).decode()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def quantize_features(features, num_bins=16):
    """
    Quantize biometric features into a binary key.
    """
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    quantized_keys = []
    for feature in normalized_features:
        bins = np.digitize(feature, np.linspace(0, 1, num_bins))
        binary_key = "".join([format(x, '04b') for x in bins])[:256]  # Truncate to 256 bits
        quantized_keys.append(binary_key)
    return quantized_keys

def binary_to_bytes(binary_key):
    """
    Convert binary string to AES key.
    """
    return int(binary_key, 2).to_bytes(32, byteorder='big')  # AES-256

def generate_biometric_features(num_samples=450, feature_size=128):
    """
    Simulate biometric features dynamically.
    """
    return np.random.rand(num_samples, feature_size)

# --- Sidebar Setup ---
st.sidebar.title("🔧 Options")
option = st.sidebar.radio("Choose Input Method:", ["Upload Biometric Features (.npy)", "Generate Biometric Features Dynamically"])
num_bins = st.sidebar.slider("Number of Bins for Quantization:", 8, 32, 16)
uploaded_file = None

# --- Main App ---
display_title()

# Step 1: Upload or Generate Features
st.subheader("🧬 Biometric Feature Selection")
if option == "Upload Biometric Features (.npy)":
    uploaded_file = st.file_uploader("Upload Combined Biometric Features (.npy):", type=["npy"])
    if uploaded_file:
        biometric_features = np.load(uploaded_file)
        st.success("Biometric Features Loaded Successfully!")
else:
    if st.button("Generate Biometric Features"):
        biometric_features = generate_biometric_features()
        st.success("Biometric Features Generated Dynamically!")

# Proceed if features are available
if 'biometric_features' in locals():
    binary_keys = quantize_features(biometric_features, num_bins=num_bins)
    st.session_state['binary_key'] = binary_keys[0]
    st.info("Biometric Key Ready for Encryption!")

# Step 2: Encryption
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
    else:
        st.error("Biometric key not available. Generate or upload features first.")

# Step 3: Decryption
st.subheader("🔓 Decrypt Your Data")
if st.button("Decrypt Data"):
    if 'encrypted_data' in st.session_state:
        aes_key = binary_to_bytes(st.session_state['binary_key'])
        decrypted_data = aes_encrypt_decrypt(st.session_state['encrypted_data'], aes_key, mode="decrypt", 
                                             nonce=st.session_state['nonce'], tag=st.session_state['tag'])
        if decrypted_data:
            st.success("Data Decrypted Successfully!")
            st.code(f"Decrypted Data: {decrypted_data}")
    else:
        st.error("No data to decrypt. Encrypt something first!")
