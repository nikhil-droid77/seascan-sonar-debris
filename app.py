import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Setup
st.set_page_config(page_title="SeaScan Sonar Debris Detector", layout="wide")
st.title("🌊 SeaScan: AI-Powered Underwater Sonar Debris Detection")
st.write("Upload a forward-looking sonar image to run real-time debris and mine detection.")

# Load Model
@st.cache_resource
def load_yolo_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_yolo_model()

# Image Upload
uploaded_file = st.file_uploader("Choose a sonar sonogram image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Uploaded Sonar Image")
        st.image(image, use_container_width=True)
    
    # Run Detection Button
    if st.button("Detect Debris & Hazards"):
        with st.spinner("Processing sonogram through YOLOv8..."):
            results = model.predict(image, conf=0.25)
            res_plotted = results[0].plot()
            
            with col2:
                st.subheader("Model Detections")
                st.image(res_plotted, use_container_width=True)
            
            # Print Detections Summary
            boxes = results[0].boxes
            st.divider()
            if len(boxes) > 0:
                st.success(f"Successfully detected {len(boxes)} target(s) in the sonogram!")
            else:
                st.warning("No debris or hazards detected above confidence threshold.")