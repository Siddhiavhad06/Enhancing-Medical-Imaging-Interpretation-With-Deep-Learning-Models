import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model for kidney stone detection
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Hp\OneDrive\Desktop\LifePredict\Model\kidney_model_resnet50.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please check the file path.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to check if the image is grayscale (Kidney scans should be grayscale)
def is_grayscale(image):
    img_array = np.array(image.convert("RGB"))
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    return np.allclose(r, g) and np.allclose(g, b)

# Function to check if the image has a circular or oval shape
def is_circular_or_oval(image):
    """Checks if the image contains a circular or oval shape."""
    img_array = np.array(image.convert("L"))  # Convert to grayscale
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=40, maxRadius=150)

    if circles is not None and len(circles) > 0:
        return True  # A valid circle was detected

    # Detect contours for oval shape detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if len(cnt) >= 5:  # Fit an ellipse only if enough points exist
            ellipse = cv2.fitEllipse(cnt)
            major_axis, minor_axis = ellipse[1]  # Get axes lengths
            
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0  # Aspect ratio check
            
            if 0.6 < aspect_ratio < 1.4:  # Ensures shape is circular or oval
                return True

    return False  # No valid shape detected

# Prediction function
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction)
    class_labels = ["Cyst", "Tumor", "Stone", "Normal"]
    return class_labels[class_idx], prediction[0][class_idx] * 100

# Streamlit UI for uploading image and prediction
st.title("Kidney Stone Detection")

uploaded_file = st.file_uploader("Upload Kidney Scan (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for display only
    display_image = image.resize((300, 300))
    st.image(display_image, caption="Uploaded Image", use_container_width=False)

    # Step 1: Validate Image Type (Grayscale & Shape Check)
    if not is_grayscale(image):
        st.error("Invalid Image! Please upload a valid kidney scan.")
    elif not is_circular_or_oval(image):
        st.error("Invalid Image! Please upload a valid kidney scan.")
    else:
        if st.button("Predict"):
            result, confidence = predict_image(image)
            st.write(f"### Prediction: {result}")
            st.write(f"### Confidence: {confidence:.2f}%")
            
            st.markdown("## Prevention Measures")
            st.write("- Stay hydrated and drink plenty of water")
            st.write("- Reduce salt and animal protein intake")
            st.write("- Maintain a healthy diet rich in calcium and fiber")
            st.write("- Avoid excessive sugar and processed foods")

            st.markdown("## Common Symptoms")
            st.write("- Severe pain in the back or side")
            st.write("- Blood in urine")
            st.write("- Frequent urination")
            st.write("- Painful urination")
            st.write("- Nausea or vomiting")
            st.write("- Fever and chills if infection is present")

else:
    st.info("Please upload a kidney scan to proceed.")
