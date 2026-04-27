# pages/breast_cancer.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

# Load trained model for breast cancer detection
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r"C:\Users\Hp\OneDrive\Desktop\LifePredict\Model\cancer_detection_model.h5"
    )  # Update with your model path
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((48, 48))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to validate if image looks like histopathology scan
def is_valid_histopathology_image(image):
    img_array = np.array(image.convert("RGB"))  # Convert to RGB NumPy array

    # Compute average color values
    avg_color = np.mean(img_array, axis=(0, 1))

    # Ensure high RGB values to match light histopathology images
    if not (avg_color[0] > 150 and avg_color[1] > 100 and avg_color[2] > 150):
        return False  # Reject images that are too dark or have incorrect colors

    # Convert to HSV (better for color filtering)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define a broader range for pink/purple histopathology stains
    lower_pink = np.array([120, 20, 70])  
    upper_pink = np.array([180, 255, 255])

    # Create mask to detect pink/purple regions
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    match_percentage = (np.sum(mask > 0) / mask.size) * 100  

    # Reject images if color match is below 15%
    if match_percentage < 15:
        return False

    # Texture Analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  

    # Check for texture complexity
    return laplacian_var > 15  # Threshold for valid image

# Prediction function
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction)
    class_labels = ["Benign (Non-Cancerous)", "Malignant (Cancerous)"]
    return class_labels[class_idx], prediction[0][class_idx] * 100

# Streamlit UI for uploading image and prediction
st.title(" Breast Cancer Detection")

uploaded_file = st.file_uploader("Upload Image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    #  Resize image for display only (small size)
    display_image = image.resize((300, 300))  # Change to (200, 200) for smaller display
    st.image(display_image, caption="Uploaded Image")  # Removed use_container_width

    # Step 1: Validate Image Type
    if not is_valid_histopathology_image(image):
        st.error(" Invalid Image! Please upload a valid histopathology scan.")
    else:
        if st.button("Predict"):
            result, confidence = predict_image(image)
            st.write(f"###  Prediction: {result}")
            st.write(f"###  Confidence: {confidence:.2f}%")

            st.markdown("## Prevention Measures")
            st.write("- Maintain a healthy weight")
            st.write("- Exercise regularly")
            st.write("- Limit alcohol consumption")
            st.write("- Avoid smoking")
            st.write("- Regular screening and check-ups")

            st.markdown("## Common Symptoms")
            st.write("- New lump in the breast or underarm")
            st.write("- Thickening or swelling of part of the breast")
            st.write("- Irritation or dimpling of breast skin")
            st.write("- Redness or flaky skin")
            st.write("- Nipple discharge other than breast milk")
            st.write("- Change in size or shape of the breast")

else:
    st.info(" Please upload an image to proceed.")
