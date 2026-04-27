# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 20:54:21 2025
@author: Hp
"""

import streamlit as st

def main():
    st.set_page_config(page_title="LifePredict", layout="wide")
    
    # Display the uploaded image in the left corner
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Screenshot 2025-03-20 160327.png", width=150)  # Adjust width as needed
    with col2:
        st.title("LifePredict: AI-Powered Disease Detection Using Deep Learning")
    
    st.write("Welcome to LifePredict. Select a disease to learn more and proceed to detection.")

    # Disease selection
    disease = st.selectbox("Select a Disease:", ["Breast Cancer", "Brain Tumor", "Kidney Disease"])

    # Display disease info (without images)
    if disease == "Breast Cancer":
        st.write("Breast cancer is a disease where cells in the breast grow uncontrollably. "
                 "Early detection increases the chances of successful treatment.")

        if st.button("Go to Breast Cancer Detection"):
            st.switch_page("pages/BREAST.py")

    elif disease == "Brain Tumor":
        st.write("A brain tumor is a mass or growth of abnormal cells in the brain. "
                 "Early detection using MRI or CT scans helps in better treatment and management.")

        if st.button("Go to Brain Tumor Detection"):
            st.switch_page("pages/BRAIN.py")

    elif disease == "Kidney Disease":
        st.write("Kidney disease affects kidney function, leading to waste buildup in the blood. "
                 "It can be diagnosed through blood tests, urine tests, or imaging scans.")

        if st.button("Go to Kidney Disease Detection"):
            st.switch_page("pages/KIDNEY.py")

if __name__ == "__main__":
    main()
