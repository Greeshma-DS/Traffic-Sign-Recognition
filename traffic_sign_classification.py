import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "traffic_sign_model.h5"
model = load_model(MODEL_PATH)

# Class labels (0–42)
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing for vehicles > 3.5 tons', 
    11:'Right-of-way at the next intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Vehicles > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve to the left', 
    20:'Dangerous curve to the right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End of all speed and passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End of no passing by vehicles > 3.5 tons'
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition")
st.markdown("Upload a traffic sign image and see the prediction!")

# -------------------------------
# Sidebar (Project Context)
# -------------------------------
with st.sidebar:
    st.header("📘 About This Project")
    st.markdown("""
    This is a **Traffic Sign Recognition System** built using:
    - Convolutional Neural Networks  
    - Keras/TensorFlow  
    - Streamlit UI  
    - German Traffic Sign Recognition Benchmark (GTSRB) dataset  

    The model predicts one of **43 traffic sign categories**.
    """)

    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. Upload an image of a traffic sign  
    2. Image is resized to **30×30**  
    3. Passed into the trained CNN model  
    4. Model outputs predicted class + confidence  
    """)

    st.markdown("### 📊 Model Info")
    st.info("Accuracy: ~96% on test data")

# -------------------------------
# Image Upload + Prediction
# -------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((30,30))
    np_img = np.array(img, dtype='float32') / 255.0
    np_img = np.expand_dims(np_img, axis=0)

    # Predict
    preds = model.predict(np_img)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    # ----------------------------------------
    # BIG CENTERED PREDICTION OUTPUT
    # ----------------------------------------
    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px;'>
            <h1 style='color:#0078ff; font-weight:700;'>
                {classes[pred_class]}
            </h1>
            <h3 style='color:#444;'>Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence Bar
    st.progress(confidence / 100)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
✨ **Developed by Greeshma**  
Traffic Sign Recognition using Deep Learning  
""")
