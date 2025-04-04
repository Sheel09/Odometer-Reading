import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
from cnocr import CnOcr
import numpy as np

# Load Models
@st.cache_resource
def load_yolo_model():
    return YOLO("runs/custom_train/weights/best.pt")  # Load YOLO Model

@st.cache_resource
def load_ocr_model():
    return CnOcr()  # Load OCR Model

# Initialize models
yolo_model = load_yolo_model()
ocr_model = load_ocr_model()

# Function to perform object detection and crop AOI
def detect_and_crop(image_path):
    # Read the image
    original_image = cv2.imread(image_path)
    results = yolo_model.predict(source=image_path, imgsz=640)
    
    cropped_images = []
    for r in results:
        boxes = r.boxes  # Detected bounding boxes
        
        for box in boxes:
            # Extract coordinates and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0])  # Class of the object
            
            # Check if the detected class is 0
            if cls == 0:
                # Crop the detected region
                cropped_img = original_image[y1:y2, x1:x2]
                cropped_images.append(cropped_img)
                
                # Save temporarily for OCR
                cv2.imwrite("temp.jpg", cropped_img)
    return cropped_images

# Function to run OCR and extract text
def run_ocr(image_path):
    ocr_output = ocr_model.ocr(image_path)
    return ''.join(item['text'] for item in ocr_output)

# Streamlit App Interface
st.title("Car Odometer Reading Detection")
st.write("### Upload an Image to Detect and Read Odometer Value")

# File uploader
uploaded_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Save uploaded image temporarily
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Display uploaded image
    st.image(Image.open(image_path), caption="Uploaded Image", use_container_width=True)
    
    # Perform detection and cropping
    st.write("### Detecting Area of Interest...")
    cropped_images = detect_and_crop(image_path)
    
    if cropped_images:
        st.write("### Cropped Region(s):")
        for idx, cropped_img in enumerate(cropped_images):
            # Convert to RGB for display
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            st.image(cropped_img_rgb, caption=f"Cropped Image {idx + 1}", use_container_width=False)
            
            # Save and run OCR
            cv2.imwrite(f"cropped_{idx}.jpg", cropped_img)
            st.write("#### Performing OCR on Cropped Image...")
            ocr_text = run_ocr(f"cropped_{idx}.jpg")
            st.write(f"**Detected Text**: {ocr_text}")
    else:
        st.write("No AOI (Area of Interest) detected. Please try another image.")
