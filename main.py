import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from Cryptodome.Cipher import AES

# Load Precomputed Features
def load_precomputed_features():
    with open("combined_features.pkl", "rb") as f:
        return pickle.load(f)

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

# Load Precomputed Features
combined_features = load_precomputed_features()

# Streamlit Frontend
st.title("🔒 Biometric Key-Based AES Encryption")
st.markdown("### A Streamlit app to demonstrate AES encryption using biometric keys.")

# Sidebar for Options
st.sidebar.title("Options")
option = st.sidebar.radio("Choose an Option:", ["Generate Key Dynamically", "Encrypt a File"])

if option == "Generate Key Dynamically":
    # Sidebar input for number of bins
    st.sidebar.subheader("Quantization Settings")
    num_bins = st.sidebar.slider("Number of Bins for Quantization:", 8, 32, 16)

    # Generate Key Dynamically
    if st.button("Generate Key Dynamically"):
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

# Footer Section
st.markdown("---")
st.markdown("© 2024 Karen Rena C. All rights reserved.")
