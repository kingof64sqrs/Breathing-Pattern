import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import librosa

# Load the trained model
model = load_model('predict.h5')

# Define the mapping of class indices to disease categories
class_mapping = {
    0: "Pneumonia",
    1: "URTI",
    2: "Healthy",
    3: "Asthma",
    4: "COPD"
}

# Self-care tips for each respiratory condition
self_care_tips = {
    "Pneumonia": ["Consult a doctor immediately", "Rest as much as possible", "Stay hydrated", "Take prescribed medications"],
    "URTI": ["Consult a doctor if symptoms persist", "Get plenty of rest", "Stay hydrated", "Use over-the-counter medications for symptom relief"],
    "Healthy": ["Maintain a healthy lifestyle with balanced diet and regular exercise", "Practice good hygiene", "Stay hydrated", "Get enough sleep"]
}

# YouTube video links for each respiratory condition
youtube_links = {
    "Pneumonia": ["https://www.youtube.com/embed/GNPb2EFOxtk", "https://www.youtube.com/embed/Wb1Q-PTaah0"],
    "URTI": ["https://www.youtube.com/embed/bcYsTeZRoVI", "https://www.youtube.com/embed/4HEDoGpWz3E"],
    "Healthy": ["https://www.youtube.com/embed/V3kZLffFxHo", "https://www.youtube.com/embed/8mvPX"]
}

# Function to extract features from audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = 862 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        st.error(f"An error occurred while parsing file {file_name}: {e}")
        return None
    return mfccs

# Streamlit app
st.title('Breathing Pattern Classification')

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the uploaded file
    features = extract_features("temp.wav")

    if features is not None:
        # Reshape features for the model input
        features = features.reshape(1, 40, 862, 1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Map the predicted class to the corresponding label
        predicted_label = class_mapping.get(predicted_class[0], "Unknown")
        
        st.write(f'Prediction: {predicted_label}')

        # Display self-care tips
        if predicted_label in self_care_tips:
            st.write("## Self-Care Tips:")
            for tip in self_care_tips[predicted_label]:
                st.write("- " + tip)
                
            # Display YouTube videos
            st.write("## Suggested YouTube Videos:")
            for youtube_link in youtube_links[predicted_label]:
                st.write("### Watch the video below:")
                st.write(f'<iframe width="560" height="315" src="{youtube_link}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
        else:
            st.write("Self-care tips not available for this condition.")
    else:
        st.error("Could not extract features from the audio file. Please try a different file.")
