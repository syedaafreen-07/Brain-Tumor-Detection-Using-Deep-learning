
import streamlit as st
from model_performance import show_model_performance
from model_prediction import show_model_prediction
from styles import load_css
import base64
load_css()

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
page=st.sidebar.radio(
    "Navigation",
    ["Home","Model Performance", "Model Prediction"]
)

def home_page():
# ---------------- PAGE STYLING ---------------- #
    st.markdown("""
    <style>
    .stApp{
        background: linear-gradient(to bottom right, #AFCFFF, #EAF4FC);
        color: #16324F;
    }

    section[data-testid="stSidebar"]{
        background: #19405c;
    }

    section[data-testid="stSidebar"] *{
        color: #DCEEFF !important;
    }

    .main-title{
        font-size: 52px;
        font-weight: bold;
        color: #16324F;
        margin-top: 40px;
    }

    .sub-text{
        font-size: 20px;
        color: #2C4E6C;
        line-height: 1.8;
    }

    .section-box{
        background-color: rgba(255,255,255,0.35);
        padding: 25px;
        border-radius: 18px;
        margin-top: 30px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    }

    .section-title{
        font-size: 32px;
        font-weight: bold;
        color: #16324F;
        margin-bottom: 15px;
    }

    .content-text{
        font-size: 18px;
        color: #244866;
        line-height: 1.9;
        text-align: justify;
    }

    .image-container{
        position: relative;
        width: 350px;
        height: 300px;
        margin-left: auto;
        margin-right: 20px;
    }

    .brain-img{
        position: absolute;
        width: 230px;
        border-radius: 18px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    }

    .img1{
        top: 0;
        left: 0;
        z-index: 1;
    }

    .img2{
        top: 80px;
        left: 120px;
        z-index: 2;
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------------- HERO SECTION ---------------- #

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown("""
        <div class="main-title">
         Brain Tumor Classification    System
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sub-text">
        An AI-powered Deep Learning application for automatic brain tumor detection and classification using MRI images.
        The system analyzes MRI scans and predicts tumor categories with the help of CNN-based models.
        </div>
        """, unsafe_allow_html=True)
       
    def get_base64(img_path):
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    img = get_base64(
        "C:/Users/syeda/OneDrive/Desktop/Project/Brain-Tumor-Program-Main-Image-1024x1024.jpg"
    )
    with right_col:
        st.markdown(f"""
        <div style="text-align:center;">
            <img src="data:image/jpg;base64,{img}"
            width="350"
            style="border-radius:20px;
                box-shadow:0px 4px 15px rgba(0,0,0,0.3);">
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------- ABOUT BRAIN TUMOR ---------------- #
    left_col, right_col = st.columns([1, 1.5])

    img = get_base64(
        "C:/Users/syeda/OneDrive/Desktop/Project/Screenshot 2026-05-25 153204.png"
    )

    with left_col:
        st.markdown(f"""
        <div class="section-box">
            <div style="text-align:center; margin-top:20px;">
                <img src="data:image/jpg;base64,{img}"
                width="320"
                style="border-radius:20px;
                    box-shadow:0px 4px 15px rgba(0,0,0,0.3);">
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div class="section-box">

        <div class="section-title">
        About Brain Tumor
        </div>

        <div class="content-text">
        A brain tumor is an abnormal growth of cells inside the brain. 
        Brain tumors can be cancerous or non-cancerous and may affect brain 
        functioning depending on their size and location.
        Early detection of brain tumors is very important for effective 
        treatment and patient care.
                    
        MRI (Magnetic Resonance Imaging) scans are widely used by doctors 
        to identify tumor regions inside the brain.
        This project uses Artificial Intelligence and Deep Learning techniques 
        to assist in automatic tumor classification from MRI images.

        The prediction system provides faster and more accurate results compared 
        to manual analysis. It also helps doctors in early diagnosis and supports
        medical decision-making.
        </div>

        </div>
        """, unsafe_allow_html=True)
    # ---------------- PREDICTION SECTION ---------------- #

    st.markdown("""
    <div class="section-box">

    <div class="section-title">
    🔍 How Prediction is Done
    </div>

    <div class="content-text">
    The uploaded MRI image first undergoes preprocessing steps such as resizing, normalization, and image enhancement.
    The processed image is then passed into trained Deep Learning models which extract important image features automatically.
    The model analyzes patterns present in the MRI scan and predicts the corresponding tumor category.
    The final prediction result is displayed along with model confidence and performance evaluation.
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- CNN SECTION ---------------- #
    img = get_base64(
    "C:/Users/syeda/OneDrive/Desktop/Project/Screenshot 2026-05-25 153204.png"
)
    st.markdown(f"""
    <div class="section-title">
    🤖 About CNN Model
    </div>

    <div class="content-text">
    CNN (Convolutional Neural Network) is a Deep Learning algorithm mainly used for image classification tasks.
    CNN automatically learns important visual features such as edges, textures, shapes, and tumor regions directly from MRI images.
    Convolutional Neural Network (CNN) based models such as VGG16, ResNet50, and Xception are commonly used for prediction because they can learn complex image patterns with high accuracy. During the training phase, the models learn from labeled MRI images containing different tumor categories. Once the model is trained, a new MRI image can be uploaded to the system, and the model predicts whether the image contains a tumor and identifies its type.
    </div>
    </div>
    """, unsafe_allow_html=True)
                     
    img = get_base64(
        "C:/Users/syeda/OneDrive/Desktop/Project/65bfec2d-724e-4fbb-a3e7-4e2001b53798.jpeg"
    )
    
    st.markdown(f"""
    <div style="text-align:center;">
        <img src="data:image/jpg;base64,{img}"
        width="800"
        height:auto;
        style="border-radius:20px;
            box-shadow:0px 4px 15px rgba(0,0,0,0.3);">
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""    
    <div class="content-text">
    The prediction system provides faster and more accurate results compared to manual analysis. It also helps doctors in early diagnosis and supports medical decision-making. Visualization techniques such as Grad-CAM can additionally highlight the affected region in the MRI image, making the prediction process more understandable and reliable.
    </div>

    </div>
    """, unsafe_allow_html=True)


if page=="Home":
    home_page()
elif page=="Model Performance":
    show_model_performance()
else:
    show_model_prediction()

